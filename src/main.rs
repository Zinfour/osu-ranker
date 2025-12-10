use {
    bzip2::read::BzDecoder,
    mimalloc::MiMalloc,
    rayon::iter::{ParallelBridge, ParallelIterator},
    serde::{Deserialize, Serialize},
    serde_json::Value,
    std::{
        collections::HashMap,
        fs::File,
        io::{BufRead, BufReader, BufWriter, Read, Write},
        sync::RwLock,
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

    // Download files.
    let files = [
        files
            .iter()
            .rev()
            .find(|z| z.contains("catch_top_10000.tar"))
            .unwrap(),
        files
            .iter()
            .rev()
            .find(|z| z.contains("taiko_top_10000.tar"))
            .unwrap(),
        files
            .iter()
            .rev()
            .find(|z| z.contains("mania_top_10000.tar"))
            .unwrap(),
        files
            .iter()
            .rev()
            .find(|z| z.contains("osu_top_10000.tar"))
            .unwrap(),
    ]
    .map(|inp_file| {
        // Download file.
        let dl = reqwest::blocking::Client::builder()
            .timeout(None)
            .build()
            .unwrap()
            .get(format!("https://data.ppy.sh/{}", inp_file))
            .send()
            .unwrap()
            .bytes()
            .unwrap()
            .to_vec();

        eprintln!("done downloading {}", inp_file);
        dl
    });

    // process files.
    eprintln!("starting processing");
    files
        .into_iter()
        .zip(["catch", "taiko", "mania", "osu"])
        .par_bridge()
        .for_each(|(in_file, mode_name)| {
            let mut archive = tar::Archive::new(BzDecoder::new(in_file.as_slice()));

            let mut raw_users_entry = None;
            let mut raw_scores_entry = None;

            for file in archive.entries().unwrap() {
                let mut file = file.unwrap();
                if file.header().path().unwrap().ends_with("sample_users.sql") {
                    let mut out = vec![];
                    file.read_to_end(&mut out).unwrap();
                    raw_users_entry = Some(out);
                } else if file.header().path().unwrap().ends_with("scores.sql") {
                    let mut out = vec![];
                    file.read_to_end(&mut out).unwrap();
                    raw_scores_entry = Some(out);
                }
            }

            let raw_users_entry = raw_users_entry.unwrap();
            let raw_scores_entry = raw_scores_entry.unwrap();

            let user_data = RwLock::new(HashMap::new());

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
                            user_data
                                .write()
                                .unwrap()
                                .insert(user_id.parse::<usize>().unwrap(), username);
                        }
                    }
                });

            let mut counted_scores = HashMap::new();
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
                                total_score,
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
                                let total_score: usize = total_score.parse().unwrap();
                                let unescaped = unescape(&data);
                                let unescaped = &unescaped[1..unescaped.len() - 1];
                                let mut data: Data = serde_json::from_str(unescaped).unwrap();
                                data.mods.sort();
                                counted_scores
                                    .entry((user_id, beatmap_id))
                                    .and_modify(
                                        |(current_total_score, current_pp, current_data)| {
                                            if pp > *current_pp {
                                                *current_total_score = total_score;
                                                *current_pp = pp;
                                                *current_data = data.clone();
                                            }
                                        },
                                    )
                                    .or_insert((total_score, pp, data));
                            }
                        }
                    }
                });

            let mut grouped_scores = HashMap::new();

            for s @ ((user_id, _), _) in &counted_scores {
                grouped_scores.entry(user_id).or_insert(vec![]).push(s);
            }

            let mut grouped_scores = grouped_scores.into_iter().collect::<Vec<_>>();

            grouped_scores.iter_mut().for_each(|z| {
                z.1.sort_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap().reverse())
            });

            loop {
                let mut map_mods = HashMap::new();
                for ((_, beatmap_id), (_, _, data)) in
                    grouped_scores.iter().flat_map(|z| z.1.iter())
                {
                    *map_mods.entry((beatmap_id, data.mods.clone())).or_insert(0) += 1;
                }
                let mut changed = false;
                for scores in grouped_scores.iter_mut() {
                    let len_before = scores.1.len();
                    scores.1.retain(|((_, beatmap_id), (_, _, data))| {
                        map_mods[&(beatmap_id, data.mods.clone())] > 1
                    });
                    if scores.1.len() != len_before {
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }

            grouped_scores.iter_mut().for_each(|z| z.1.truncate(100));

            loop {
                let mut map_mods = HashMap::new();
                for ((_, beatmap_id), (_, _, data)) in
                    grouped_scores.iter().flat_map(|z| z.1.iter())
                {
                    *map_mods.entry((beatmap_id, data.mods.clone())).or_insert(0) += 1;
                }
                let mut changed = false;
                for scores in grouped_scores.iter_mut() {
                    let len_before = scores.1.len();
                    scores.1.retain(|((_, beatmap_id), (_, _, data))| {
                        map_mods[&(beatmap_id, data.mods.clone())] > 1
                    });
                    if scores.1.len() != len_before {
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }

            let mut output = vec![];
            for ((user_id, beatmap_id), (total_score, _, data)) in
                grouped_scores.into_iter().flat_map(|z| z.1.into_iter())
            {
                output.push((user_id, beatmap_id, data.mods.clone(), total_score));
            }

            let mut file = BufWriter::new(
                File::create(format!("./processed_score_files/{}.json", mode_name)).unwrap(),
            );
            serde_json::to_writer(&mut file, &output).unwrap();
            file.flush().unwrap();

            let mut file = BufWriter::new(
                File::create(format!("./processed_score_files/{}_users.csv", mode_name)).unwrap(),
            );
            for (k, v) in user_data.into_inner().unwrap().into_iter() {
                writeln!(&mut file, "{},{}", k, v).unwrap();
            }
            file.flush().unwrap();
        });
}
