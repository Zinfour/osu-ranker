use {
    bzip2::read::BzDecoder,
    mimalloc::MiMalloc,
    rayon::iter::{ParallelBridge, ParallelIterator},
    serde::{Deserialize, Serialize},
    serde_json::Value,
    std::{
        cmp::Ordering,
        collections::{BinaryHeap, HashMap, HashSet},
        fs::File,
        io::{BufRead, BufReader, BufWriter, Read, Write},
    },
};

// MiMalloc is usually faster than the default allocator on windows.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
struct Mod {
    acronym: String,
    settings: Option<HashMap<String, Value>>,
}

impl std::hash::Hash for Mod {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.acronym.hash(state);
    }
}

impl Ord for Mod {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.acronym.cmp(&other.acronym)
    }
}

impl PartialOrd for Mod {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
struct Data {
    mods: Vec<Mod>,
    #[serde(skip_deserializing)]
    statistics: Value,
    #[serde(skip_deserializing)]
    maximum_statistics: Value,
}

fn unescape(st: &str) -> String {
    let mut out = String::new();
    let mut escaping = false;
    for c in st.chars() {
        if c != '\\' || escaping {
            out.push(c);
        }
        escaping = c == '\\';
    }
    out
}
#[derive(PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

fn get_top_eligible(
    mm: (usize, usize),
    n: usize,
    scan_pos: &HashMap<(usize, usize), usize>,
    mapmod_to_users: &HashMap<(usize, usize), Vec<(usize, f32)>>,
    user_remaining: &HashMap<usize, usize>,
    user_used_maps: &HashMap<usize, HashSet<usize>>,
) -> Vec<(usize, f32)> {
    let mut result = Vec::new();
    for &(uid, pp) in &mapmod_to_users.get(&mm).unwrap()[*scan_pos.get(&mm).unwrap()..] {
        if user_remaining[&uid] == 0 {
            continue;
        }
        if user_used_maps.get(&uid).is_some_and(|s| s.contains(&mm.0)) {
            continue;
        }
        result.push((uid, pp));
        if result.len() == n {
            break;
        }
    }
    result
}

pub fn solve_greedy(
    scores: HashMap<(usize, usize, usize), f32>,
    max_per_user: usize,
    mode_name: &str,
) -> HashMap<(usize, usize, usize), f32> {
    // Build indexes
    let mut mapmod_to_users: HashMap<(usize, usize), Vec<(usize, f32)>> = HashMap::new();
    let mut user_remaining: HashMap<usize, usize> = HashMap::new();

    for ((user_id, beatmap_id, mod_id), pp) in scores {
        mapmod_to_users
            .entry((beatmap_id, mod_id))
            .or_default()
            .push((user_id, pp));
        user_remaining.insert(user_id, max_per_user);
    }

    for users in mapmod_to_users.values_mut() {
        users.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    }

    // Remove singletons
    let singletons: Vec<(usize, usize)> = mapmod_to_users
        .iter()
        .filter(|(_, uids)| uids.len() == 1)
        .map(|(mm, _)| *mm)
        .collect();
    for mm in singletons {
        mapmod_to_users.remove(&mm);
    }

    // Greedy state
    let mut user_used_maps = HashMap::new();
    let mut active_mapmods = HashSet::new();
    let mut selected = HashMap::new();
    let mut scan_pos = mapmod_to_users.keys().map(|&mm| (mm, 0)).collect();
    let mut counter = 0;

    let mut heap: BinaryHeap<_> = mapmod_to_users
        .keys()
        .filter_map(|&mm| {
            let top = get_top_eligible(
                mm,
                2,
                &scan_pos,
                &mapmod_to_users,
                &user_remaining,
                &user_used_maps,
            );
            if top.len() == 2 {
                let avg = (top[0].1 + top[1].1) / 2.0;
                let c = counter;
                counter += 1;
                Some((OrdF32(avg), std::cmp::Reverse(c), mm))
            } else {
                None
            }
        })
        .collect();

    while let Some((OrdF32(popped_pri), _, mm)) = heap.pop() {
        let (mid, mod_id) = mm;
        let is_active = active_mapmods.contains(&mm);
        let n = if is_active { 1 } else { 2 };

        let top = get_top_eligible(
            mm,
            n,
            &scan_pos,
            &mapmod_to_users,
            &user_remaining,
            &user_used_maps,
        );
        if top.len() < n {
            continue;
        }

        let actual_pri = if is_active {
            top[0].1
        } else {
            (top[0].1 + top[1].1) / 2.0
        };

        if actual_pri < popped_pri {
            heap.push((OrdF32(actual_pri), std::cmp::Reverse(counter), mm));
            counter += 1;
            continue;
        }

        for &(uid, pp) in &top {
            let remaining = user_remaining.get_mut(&uid).unwrap();
            assert!(*remaining > 0);
            selected.insert((uid, mid, mod_id), pp);
            *remaining -= 1;
            user_used_maps.entry(uid).or_default().insert(mid);
        }

        if !is_active {
            active_mapmods.insert(mm);
        }

        // Advance scan_pos past consumed users
        if let Some(users) = mapmod_to_users.get(&mm) {
            let pos = scan_pos.get_mut(&mm).unwrap();
            while *pos < users.len() {
                let uid = users[*pos].0;
                if user_remaining[&uid] > 0
                    && !user_used_maps.get(&uid).is_some_and(|s| s.contains(&mid))
                {
                    break;
                }
                *pos += 1;
            }
        }

        // Re-push with n=1 since mapmod is now active
        let top_next = get_top_eligible(
            mm,
            1,
            &scan_pos,
            &mapmod_to_users,
            &user_remaining,
            &user_used_maps,
        );
        if !top_next.is_empty() {
            heap.push((OrdF32(top_next[0].1), std::cmp::Reverse(counter), mm));
            counter += 1;
        }
    }

    // Sanity checks
    let mut user_counts: HashMap<usize, usize> = HashMap::new();
    let mut mm_counts: HashMap<(usize, usize), usize> = HashMap::new();
    let mut user_map_counts: HashMap<(usize, usize), usize> = HashMap::new();

    for (uid, mid, mod_id) in selected.keys() {
        *user_counts.entry(*uid).or_default() += 1;
        *mm_counts.entry((*mid, *mod_id)).or_default() += 1;
        *user_map_counts.entry((*uid, *mid)).or_default() += 1;
    }

    assert!(
        user_counts.iter().all(|(_, c)| *c <= max_per_user),
        "User cap violated!"
    );
    assert!(
        mm_counts.values().all(|&c| c >= 2),
        "Map/mod constraint violated!"
    );
    assert!(
        user_map_counts.values().all(|&c| c == 1),
        "(user, map) uniqueness violated!"
    );

    eprintln!("{}: Sanity checks passed.", mode_name);
    eprintln!(
        "{}: Active (map, mod) pairs: {}",
        mode_name,
        active_mapmods.len()
    );
    eprintln!("{}: Total scores: {}", mode_name, selected.len());

    selected
}

fn main() {
    // Download file list.
    let website = reqwest::blocking::get("https://data.ppy.sh/")
        .unwrap()
        .text()
        .unwrap();

    let mut files = vec![];

    let mut current_position = 0;
    while let Some(matched) = website[current_position..].find("href='") {
        current_position += matched + "href='".len();
        files.push(
            &website[current_position
                ..current_position + website[current_position..].find('\'').unwrap()],
        );
    }

    ["catch", "taiko", "mania", "osu"]
        .into_iter()
        .map(|mode_name| {
            let inp_file = files
                .iter()
                .rev()
                .find(|z| z.contains(&format!("{}_top_10000.tar", mode_name)))
                .unwrap();


            
            eprintln!("starting downloading {}", inp_file);
            let response = reqwest::blocking::Client::builder()
                .timeout(None)
                .build()
                .unwrap()
                .get(format!("https://data.ppy.sh/{}", inp_file))
                .send()
                .unwrap();

            let mut archive = tar::Archive::new(BzDecoder::new(BufReader::new(response)));

            let mut raw_users_entry = vec![];
            let mut raw_scores_entry = vec![];

            for file in archive.entries().unwrap() {
                let mut file = file.unwrap();
                if file.header().path().unwrap().ends_with("sample_users.sql") {
                    file.read_to_end(&mut raw_users_entry).unwrap();
                } else if file.header().path().unwrap().ends_with("scores.sql") {
                    file.read_to_end(&mut raw_scores_entry).unwrap();
                }
            }
            eprintln!("done downloading and extracting {}", inp_file);
            (raw_users_entry, raw_scores_entry, mode_name)
        })
        .par_bridge()
        .for_each(|(raw_users_entry, raw_scores_entry, mode_name)| {
            let mut user_data = HashMap::new();

            BufReader::new(raw_users_entry.as_slice())
                .lines()
                .map(|l| l.unwrap())
                .for_each(|line: String| {
                    if let Some(stripped) = line.strip_prefix("INSERT INTO `sample_users` VALUES (")
                    {
                        let mut b_line = stripped.as_bytes();
                        while b_line != [b';'] {
                            let mut slices = vec![];
                            for _ in 0..1 {
                                let mut cut_len = 0;
                                for x in b_line {
                                    if *x == b',' || *x == b')' {
                                        break;
                                    }
                                    cut_len += 1;
                                }
                                slices.push(&b_line[..cut_len]);
                                b_line = &b_line[cut_len + 1..];
                            }

                            let mut cut_len = 0;
                            let mut escaping = false;
                            for x in b_line {
                                if !escaping && *x == b',' {
                                    break;
                                }
                                escaping = *x == b'\\';
                                cut_len += 1;
                            }
                            slices.push(&b_line[..cut_len]);
                            b_line = &b_line[cut_len + 1..];

                            for _ in 0..2 {
                                let mut cut_len = 0;
                                for x in b_line {
                                    if *x == b',' || *x == b')' {
                                        break;
                                    }
                                    cut_len += 1;
                                }
                                slices.push(&b_line[..cut_len]);
                                b_line = &b_line[cut_len + 1..];
                            }
                            if b_line.len() >= 3 && &b_line[..2] == b",(" {
                                b_line = &b_line[",(".len()..];
                            }

                            let tmp: [&[u8]; 4] = slices.try_into().unwrap();
                            let [user_id, username, _user_warnings, _user_type] =
                                tmp.map(|z| String::from_utf8(z.to_vec()).unwrap());
                            user_data.insert(user_id.parse::<usize>().unwrap(), username);
                        }
                    }
                });

            eprintln!("parsed user data for {}", mode_name);

            let mut counted_scores = HashMap::new();
            let mut mods = HashMap::new();
            BufReader::new(raw_scores_entry.as_slice())
                .lines()
                .map(|l| l.unwrap())
                .for_each(|line: String| {
                    if let Some(stripped) = line.strip_prefix("INSERT INTO `scores` VALUES (") {
                        let mut b_line = stripped.as_bytes();
                        while b_line != [b';'] {
                            let mut slices = vec![];
                            for _ in 0..12 {
                                let mut cut_len = 0;
                                for x in b_line {
                                    if *x == b',' || *x == b')' {
                                        break;
                                    }
                                    cut_len += 1;
                                }
                                slices.push(&b_line[..cut_len]);
                                b_line = &b_line[cut_len + 1..];
                            }

                            let mut depth = 0;
                            let mut cut_len = 0;
                            for x in b_line {
                                if *x == b',' && depth == 0 {
                                    break;
                                } else if *x == b'(' || *x == b'[' || *x == b'{' {
                                    depth += 1
                                } else if *x == b')' || *x == b']' || *x == b'}' {
                                    depth -= 1
                                }
                                cut_len += 1;
                            }
                            slices.push(&b_line[..cut_len]);
                            b_line = &b_line[cut_len + 1..];

                            for _ in 0..7 {
                                let mut cut_len = 0;
                                for x in b_line {
                                    if *x == b',' || *x == b')' {
                                        break;
                                    }
                                    cut_len += 1;
                                }
                                slices.push(&b_line[..cut_len]);
                                b_line = &b_line[cut_len + 1..];
                            }
                            if b_line.len() >= 3 && &b_line[..2] == b",(" {
                                b_line = &b_line[",(".len()..];
                            }

                            let tmp: [&[u8]; 20] = slices.try_into().unwrap();
                            let [
                                _id,
                                user_id,
                                _ruleset_id,
                                beatmap_id,
                                _has_replay,
                                _preserve,
                                ranked,
                                _rank,
                                _passed,
                                _accuracy,
                                _max_combo,
                                _total_score,
                                data,
                                pp,
                                _legacy_score_id,
                                _legacy_total_score,
                                _started_at,
                                _ended_at,
                                _unix_updated_at,
                                _build_id,
                            ] = tmp.map(|z| String::from_utf8(z.to_vec()).unwrap());

                            if let Ok(pp) = pp.parse::<f32>()
                                && ranked == "1"
                            {
                                let user_id: usize = user_id.parse().unwrap();
                                let beatmap_id: usize = beatmap_id.parse().unwrap();
                                let unescaped_data = unescape(&data);
                                let unescaped_data = &unescaped_data[1..unescaped_data.len() - 1];
                                let mut data: Data = serde_json::from_str(unescaped_data).unwrap();
                                data.mods.sort();
                                let mod_id = {
                                    let mods_len = mods.len();
                                    *mods.entry(data.mods).or_insert(mods_len)
                                };
                                counted_scores
                                    .entry((user_id, beatmap_id, mod_id))
                                    .and_modify(|current_pp| {
                                        if pp > *current_pp {
                                            *current_pp = pp
                                        }
                                    })
                                    .or_insert(pp);
                            }
                        }
                    }
                });
            eprintln!("parsed score data for {}", mode_name);

            let chosen_scores = solve_greedy(counted_scores, 100, mode_name);

            let chosen_scores = chosen_scores
                .into_iter()
                .map(|((a, b, c), d)| (a, b, c, d))
                .collect::<Vec<_>>();

            let mut file = BufWriter::new(
                File::create(format!("./processed_score_files/{}.json", mode_name)).unwrap(),
            );
            serde_json::to_writer(&mut file, &chosen_scores).unwrap();
            file.flush().unwrap();

            let mut score_count: HashMap<_, usize> = HashMap::new();
            for (user_id, _, _, _) in chosen_scores {
                *score_count.entry(user_id).or_default() += 1;
            }
            let mut file = BufWriter::new(
                File::create(format!("./processed_score_files/{}_users.csv", mode_name)).unwrap(),
            );
            for (k, v) in user_data.into_iter() {
                writeln!(
                    &mut file,
                    "{},{},{}",
                    k,
                    v,
                    score_count.get(&k).copied().unwrap_or(0)
                )
                .unwrap();
            }
            file.flush().unwrap();

            eprintln!("{} done", mode_name);
        });
}
