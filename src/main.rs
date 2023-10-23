use markov_strings::*;
use twitch_irc::login::{CredentialsPair, StaticLoginCredentials};
use twitch_irc::message::{PrivmsgMessage, ServerMessage};
use twitch_irc::{ClientConfig, SecureTCPTransport, TwitchIRCClient};

use log::info;
use serde_json;
use std::collections::HashMap;
use std::ffi::OsString;
use std::fs;
use std::io::Write;

static mut MIN_REFS: usize = 3;
static mut MIN_LEN: usize = 120;
static mut MAX_LEN: usize = 460;
static mut MIN_SCORE: u16 = 15;
static mut MAX_TRIES: u16 = 20000;
static mut A_TILL_PROC: usize = 420;
static mut USE_FILTER: bool = false;
static mut FILTER: String = String::new();
static mut SECRET: usize = 0;

unsafe fn update_markov(m: &mut Markov) {
    m.set_state_size(2).unwrap();
    m.set_filter(|r: &MarkovResult| {
        // A minimal relative score and number of references
        // The thresholds are relative to your input
        r.score >= MIN_SCORE && r.refs.len() >= MIN_REFS
            // We want to generate random messages
            && r.text.len() >= MIN_LEN.clamp(20, MAX_LEN)
            && r.text.len() <= MAX_LEN.clamp(MIN_LEN, 460)
            // No commands
            && !r.text.starts_with('!')
            // No mentions
            // && !r.text.contains('@')
            && (!USE_FILTER || r.text.contains(&FILTER))
    })
    .set_max_tries(MAX_TRIES.clamp(10, 65000));
}

#[derive(Default)]
struct Sglypa {
    twitch_name: String,
    twitch_token: String,
    client: Option<TwitchIRCClient<SecureTCPTransport, StaticLoginCredentials>>,
    replying: bool,
    markov: Option<Markov>,
    personal_markov: Option<HashMap<String, Markov>>,
    train_data: Option<Vec<InputData>>,
    personal_train_data: Option<HashMap<String, Vec<InputData>>>,
}

impl Sglypa {
    pub fn new(twitch_name: String, twitch_token: String) -> Sglypa {
        Self {
            twitch_name,
            twitch_token,
            ..Default::default()
        }
    }

    pub fn train_from_vods(&mut self, personal: bool, vod_filter: Option<fn(&OsString) -> bool>) {
        unsafe {
            update_markov(self.markov.insert(Markov::new()));
        }
        let _ = self.train_data.insert(Vec::<InputData>::new());

        let streamers = fs::read_dir("./vods").unwrap();
        for streamer_entry in streamers {
            let streamer = streamer_entry.unwrap().path();
            if let Ok(vods) = fs::read_dir(&streamer) {
                for vod in vods {
                    let path = vod.unwrap().path();
                    if vod_filter.is_some()
                        && !path
                            .components()
                            .map(|c| c.as_os_str())
                            .any(|c| vod_filter.unwrap()(&c.to_owned()))
                    {
                        continue;
                    }
                    let json: serde_json::Value =
                        serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
                    if let serde_json::Value::Array(comments) = &json["comments"] {
                        for comment in comments.iter() {
                            if let (
                                serde_json::Value::String(body),
                                serde_json::Value::String(name_),
                            ) = (&comment["message"]["body"], &comment["commenter"]["name"])
                            {
                                let name = name_.to_lowercase();
                                if body.contains("Tier")
                                    || body.to_lowercase().contains("sglypa")
                                    || body.starts_with("!")
                                    || body.split(" ").count() < 2
                                {
                                    continue;
                                }
                                self.train_data.as_mut().unwrap().push(InputData {
                                    text: body.to_owned(),
                                    meta: None,
                                });
                                if personal {
                                    if !self.personal_markov.as_mut().unwrap().contains_key(&name) {
                                        let mut m = Markov::new();
                                        unsafe {
                                            update_markov(&mut m);
                                        }
                                        self.personal_markov
                                            .as_mut()
                                            .unwrap()
                                            .insert(name.to_owned(), m);
                                        self.personal_train_data
                                            .as_mut()
                                            .unwrap()
                                            .insert(name.to_owned(), Vec::new());
                                    }
                                    self.personal_train_data
                                        .as_mut()
                                        .unwrap()
                                        .get_mut(&name)
                                        .unwrap()
                                        .push(InputData {
                                            text: body.to_owned(),
                                            meta: None,
                                        });
                                }
                            }
                        }
                        info!("{}, {} messages", path.display(), comments.len());
                    }
                }
            }
        }
        info!(
            "{} total training messages",
            self.train_data.as_mut().unwrap().len()
        );
        self.train_data.as_mut().unwrap().sort();
        self.train_data.as_mut().unwrap().dedup();
        info!(
            "{} deduped training messages",
            self.train_data.as_mut().unwrap().len()
        );
        self.markov
            .as_mut()
            .unwrap()
            .add_to_corpus(self.train_data.as_mut().unwrap().clone());
        info!("trained main");
        if personal {
            for (name, data) in self.personal_train_data.as_mut().unwrap().iter_mut() {
                data.sort();
                data.dedup();
                self.personal_markov
                    .as_mut()
                    .unwrap()
                    .get_mut(name)
                    .unwrap()
                    .add_to_corpus(data.clone());
            }
        }
    }

    pub fn join(&mut self, channel: &str) {
        if let Some(client) = self.client.as_mut() {
            if client.join(channel.to_owned()).is_ok() {
                info!("joined {}", channel);
            } else {
                info!("failed to join {}", channel);
            }
        }
    }

    pub async fn run(&mut self) {
        let config = ClientConfig {
            login_credentials: StaticLoginCredentials {
                credentials: CredentialsPair {
                    login: self.twitch_name.to_owned(),
                    token: Some(self.twitch_token.replacen("oauth:", "", 1)),
                },
            },
            ..ClientConfig::default()
        };
        let (mut incoming_messages, client) =
            TwitchIRCClient::<SecureTCPTransport, StaticLoginCredentials>::new(config);
        let _ = self.client.insert(client);
        self.join(&self.twitch_name.to_owned());
        while let Some(message) = incoming_messages.recv().await {
            match message {
                ServerMessage::Privmsg(msg) => {
                    unsafe {
                        self.handle_msg(&msg).await;
                    }
                    info!(
                        "(#{}) {}: {}",
                        msg.channel_login, msg.sender.name, msg.message_text
                    );
                }
                ServerMessage::Whisper(msg) => {
                    info!("(w) {}: {}", msg.sender.name, msg.message_text);
                }
                _ => {}
            }
        }
    }

    pub async unsafe fn handle_msg(&mut self, msg: &PrivmsgMessage) {
        if self.replying && msg.message_text.starts_with("!info") {
            self.handle_command_info(msg).await;
        }
        // `markov.add_to_corpus(vec![InputData {
        // `    text: msg.message_text.to_owned(),
        // `    meta: None,
        // `}]);
        // if SECRET == 0 {
        //     if let Some(name) = msg
        //         .message_text
        //         .strip_prefix("!")
        //         .and_then(|n| Some(n.to_lowercase()))
        //     {
        //         if personal_markov.contains_key(&name) {
        //             if let Ok(result) = personal_markov.get(&name).unwrap().generate() {
        //                 println!("{:?}", result);
        //                 client
        //                     .say(
        //                         channel_to_join.clone(),
        //                         format!("Sglypa: {}", &result.text.to_owned()),
        //                     )
        //                     .await
        //                     .unwrap();
        //             }
        //         }
        //     }
        //     if msg.message_text.starts_with("!info") {
        //         let tokens: Vec<String> =
        //             msg.message_text.split(' ').map(|s| s.to_owned()).collect();
        //         if tokens.len() > 1 {
        //             client
        //                 .say(
        //                     channel_to_join.clone(),
        //                     format!(
        //                         "Sglypa: found {} relevant messages from {}",
        //                         personal_info.get(&tokens[1].to_lowercase()).unwrap_or(&0),
        //                         tokens[1],
        //                     ),
        //                 )
        //                 .await
        //                 .unwrap();
        //         } else {
        //             let mut top = personal_info.iter().collect::<Vec<(&String, &usize)>>();
        //             top.sort_by_key(|(_k, v)| *v);
        //             top.reverse();
        //             println!("{:?}", top);
        //             let ans = top
        //                 .iter()
        //                 .take(10)
        //                 .map(|(k, v)| format!("{}: {}\t", k, v))
        //                 .collect::<String>();
        //             println!("{}", ans);
        //             client
        //                 .say(
        //                     channel_to_join.clone(),
        //                     format!("Top 10 chatters:\t{}", ans),
        //                 )
        //                 .await
        //                 .unwrap();
        //         }
        //     }
        // }
        // if msg.message_text.starts_with("!stair") && msg.sender.login.eq("gosuto_botto")
        // //ゴーストボット")
        // {
        //     let tokens: Vec<String> = msg.message_text.split(' ').map(|s| s.to_owned()).collect();
        //     if let Ok(l) = tokens[1].parse() {
        //         let message = tokens
        //             .into_iter()
        //             .skip(2)
        //             .map(|s| s + " ")
        //             .collect::<String>();
        //         for i in (1..l).chain((1..=l).rev()) {
        //             client
        //                 .say(channel_to_join.clone(), format!("{}", message.repeat(i)))
        //                 .await
        //                 .unwrap();
        //             tokio::time::sleep(tokio::time::Duration::from_millis(35)).await;
        //         }
        //     }
        // }
        // if msg.message_text.starts_with("!sglypa") && msg.sender.login.eq("gosuto_botto")
        // //ゴーストボット")
        // {
        //     let tokens: Vec<String> = msg.message_text.split(' ').map(|s| s.to_owned()).collect();
        //     for i in 1..tokens.len() - 1 {
        //         match tokens[i].as_ref() {
        //             "MIN_REFS" => {
        //                 if let Ok(val) = tokens[i + 1].parse() {
        //                     MIN_REFS = val;
        //                 }
        //             }
        //             "MIN_LEN" => {
        //                 if let Ok(val) = tokens[i + 1].parse() {
        //                     MIN_LEN = val;
        //                 }
        //             }
        //             "MAX_LEN" => {
        //                 if let Ok(val) = tokens[i + 1].parse() {
        //                     MAX_LEN = val;
        //                 }
        //             }
        //             "MIN_SCORE" => {
        //                 if let Ok(val) = tokens[i + 1].parse() {
        //                     MIN_SCORE = val;
        //                 }
        //             }
        //             "MAX_TRIES" => {
        //                 if let Ok(val) = tokens[i + 1].parse() {
        //                     MAX_TRIES = val;
        //                 }
        //             }
        //             "PROC" => {
        //                 if let Ok(val) = tokens[i + 1].parse() {
        //                     A_TILL_PROC = val;
        //                 }
        //             }
        //             "SECRET" => {
        //                 if let Ok(val) = tokens[i + 1].parse() {
        //                     SECRET = val;
        //                 }
        //             }
        //
        //             _ => {}
        //         }
        //     }
        // }
        // if SECRET == 0
        //     && (thread_rng().gen_range(0..A_TILL_PROC) == 0 || msg.message_text.eq("!sglypa"))
        // {
        //     if let Ok(result) = self.markov.unwrap().generate() {
        //         println!("{:?}", result);
        //         self.client
        //             .unwrap()
        //             .say(
        //                 msg.channel_login,
        //                 format!("Sglypa: {}", &result.text.to_owned()),
        //             )
        //             .await
        //             .unwrap();
        //     }
        // }
    }

    pub async fn handle_command_info(&mut self, msg: &PrivmsgMessage) {
        let tokens: Vec<String> = msg.message_text.split(' ').map(|s| s.to_owned()).collect();
        if tokens.len() > 1 {
            self.client
                .as_mut()
                .unwrap()
                .say(
                    msg.channel_login.to_owned(),
                    format!(
                        "Sglypa: found {} relevant messages from {}",
                        self.personal_train_data
                            .as_mut()
                            .map(|d| d
                                .get(&tokens[1].to_lowercase())
                                .map(|arr| arr.len())
                                .unwrap_or(0))
                            .unwrap_or(0),
                        &tokens[1],
                    ),
                )
                .await
                .unwrap();
        } else if self.personal_train_data.is_some() {
            let mut top = self
                .personal_train_data
                .as_mut()
                .unwrap()
                .iter()
                .map(|(k, v)| (k, v.len()))
                .collect::<Vec<(&String, usize)>>();
            top.sort_by_key(|(_k, v)| *v);
            top.reverse();
            let ans = top
                .iter()
                .take(20)
                .map(|(k, v)| format!("{}: {};\t", k, v))
                .collect::<String>();
            self.client
                .as_mut()
                .unwrap()
                .say(
                    msg.channel_login.to_owned(),
                    format!("Top 20 chatters:\t{}", ans),
                )
                .await
                .unwrap();
        }
    }
}

#[tokio::main]
pub async fn main() {
    env_logger::Builder::new()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] - {}",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .filter(None, log::LevelFilter::Info)
        .init();

    let twitch_name = "gosuto_botto".to_owned();
    let twitch_token = "r7sgsai3jqws965ihkiujw6wpbld6s".to_owned();

    let mut sglypa = Sglypa::new(twitch_name, twitch_token);
    sglypa.train_from_vods(false, Some(|s: &OsString| s.eq("red_ponda")));
    let handle = tokio::spawn(async move {
        sglypa.run().await;
    });
    handle.await.unwrap();
    // let config = ClientConfig {
    //     login_credentials: StaticLoginCredentials {
    //         credentials: CredentialsPair {
    //             login: twitch_name.to_owned(),
    //             token: Some(twitch_token.replacen("oauth:", "", 1)),
    //         },
    //     },
    //     ..ClientConfig::default()
    // };
    // let (mut incoming_messages, client) =
    //     TwitchIRCClient::<SecureTCPTransport, StaticLoginCredentials>::new(config);
    //
    // client.join("gosuto_botto".to_owned()).unwrap();
    // while let Some(message) = incoming_messages.recv().await {
    //     match message {
    //         ServerMessage::Privmsg(msg) => {
    //             info!(
    //                 "(#{}) {}: {}",
    //                 msg.channel_login, msg.sender.name, msg.message_text
    //             );
    //         }
    //         ServerMessage::Whisper(msg) => {
    //             info!("(w) {}: {}", msg.sender.name, msg.message_text);
    //         }
    //         _ => {}
    //     }
    // }
}
