use markov_strings::*;
use twitch_irc::login::{CredentialsPair, StaticLoginCredentials};
use twitch_irc::message::{PrivmsgMessage, ServerMessage};
use twitch_irc::{ClientConfig, SecureTCPTransport, TwitchIRCClient};

use log::info;
use rand::prelude::*;
use serde_json;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

const STATE_SIZE: usize = 2;
static mut MIN_REFS: usize = 3;
static mut MIN_LEN: usize = 120;
static mut MAX_LEN: usize = 460;
static mut MIN_SCORE: u16 = 15;
static mut MAX_TRIES: u16 = 20000;
static mut A_TILL_PROC: usize = 420;
static mut USE_FILTER: bool = false;
static mut SHOW_REFS: bool = false;
static mut FILTER: String = String::new();

unsafe fn update_markov(m: &mut Markov) {
    m.set_state_size(STATE_SIZE).unwrap();
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

struct Sglypa {
    owners: HashSet<String>,
    moderators: HashSet<String>,
    incoming_messages: tokio::sync::mpsc::UnboundedReceiver<ServerMessage>,
    client: TwitchIRCClient<SecureTCPTransport, StaticLoginCredentials>,
    replying: bool,
    markov: Option<Markov>,
    nmarkov: Option<Markov>,
    personal_markov: Option<HashMap<String, Markov>>,
    train_data: Option<Vec<InputData>>,
    ntrain_data: Option<Vec<InputData>>,
    personal_train_data: Option<HashMap<String, Vec<InputData>>>,
}

impl Sglypa {
    pub fn new(twitch_name: String, twitch_token: String) -> Sglypa {
        let config = ClientConfig {
            login_credentials: StaticLoginCredentials {
                credentials: CredentialsPair {
                    login: twitch_name.to_owned(),
                    token: Some(twitch_token.replacen("oauth:", "", 1)),
                },
            },
            ..ClientConfig::default()
        };
        let (incoming_messages, client) =
            TwitchIRCClient::<SecureTCPTransport, StaticLoginCredentials>::new(config);
        let mut host = HashSet::<String>::new();
        host.insert(twitch_name.to_owned());
        Self {
            owners: host.clone(),
            moderators: host.clone(),
            incoming_messages,
            client,
            replying: false,
            markov: None,
            nmarkov: None,
            personal_markov: None,
            train_data: None,
            ntrain_data: None,
            personal_train_data: None,
        }
    }

    pub fn train_from_vods(
        &mut self,
        personal: bool,
        vod_filter: Option<fn(&PathBuf) -> bool>,
        message_filter: Option<fn(String, String) -> bool>,
    ) {
        unsafe {
            update_markov(self.markov.insert(Markov::new()));
        }
        let _ = self.train_data.insert(Vec::<InputData>::new());
        if personal {
            let _ = self.personal_markov.insert(HashMap::new());
            let _ = self.personal_train_data.insert(HashMap::new());
        }

        let streamers = fs::read_dir("./vods").unwrap();
        for streamer_entry in streamers {
            let streamer = streamer_entry.unwrap().path();
            if let Ok(vods) = fs::read_dir(&streamer) {
                for vod in vods {
                    let path = vod.unwrap().path();
                    if vod_filter.is_some() && !vod_filter.unwrap()(&path) {
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
                                if message_filter.is_some()
                                    && !message_filter.unwrap()(name.to_owned(), body.to_owned())
                                {
                                    continue;
                                }
                                self.train_data.as_mut().unwrap().push(InputData {
                                    text: body.to_owned(),
                                    meta: Some(name.to_owned()),
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
                                            meta: Some(name.to_owned()),
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
        if self.client.join(channel.to_owned()).is_ok() {
            info!("joined {}", channel);
        } else {
            info!("failed to join {}", channel);
        }
    }

    pub fn leave(&mut self, channel: &str) {
        self.client.part(channel.to_owned());
        info!("left {}", channel);
    }

    pub async fn run(&mut self) {
        while let Some(message) = self.incoming_messages.recv().await {
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

    pub async fn say_stair(&mut self, channel: &str, length: usize, message: &str) {
        for i in (1..length).chain((1..=length).rev()) {
            self.client
                .say(
                    channel.to_owned(),
                    format!("{}", (message.to_owned() + " ").repeat(i).trim_end()),
                )
                .await
                .unwrap();
            tokio::time::sleep(tokio::time::Duration::from_millis(35)).await;
        }
    }

    pub fn is_super_privileged(&self, msg: &PrivmsgMessage) -> bool {
        self.owners.contains(&msg.sender.login)
    }

    pub fn is_privileged(&self, msg: &PrivmsgMessage) -> bool {
        self.moderators.contains(&msg.sender.login) || self.owners.contains(&msg.sender.login)
    }

    pub async unsafe fn handle_msg(&mut self, msg: &PrivmsgMessage) {
        self.handle_learn(msg);
        if msg.message_text.to_lowercase().starts_with("!st") && self.is_privileged(msg) {
            self.handle_command_stair(msg, None).await;
            return;
        }
        if msg.message_text.to_lowercase().starts_with("!<>") && self.is_super_privileged(msg) {
            self.handle_command_stair(
                msg,
                msg.message_text
                    .split(" ")
                    .take(1)
                    .collect::<String>()
                    .strip_prefix("!<>"),
            )
            .await;
            return;
        }
        if self.is_privileged(msg) && msg.message_text.to_lowercase().trim().eq("!on") {
            self.replying = true;
            return;
        }
        if self.is_privileged(msg) && msg.message_text.to_lowercase().trim().eq("!off") {
            self.replying = false;
            return;
        }
        if self.is_super_privileged(msg)
            && msg.message_text.to_lowercase().trim().starts_with("!join")
        {
            let name: String = msg
                .message_text
                .split(' ')
                .skip(1)
                .take(1)
                .map(|s| s.to_owned())
                .collect::<String>();
            if !name.is_empty() {
                self.join(&name);
            }
            return;
        }
        if self.is_super_privileged(msg)
            && msg.message_text.to_lowercase().trim().starts_with("!leave")
        {
            let name: String = msg
                .message_text
                .split(' ')
                .skip(1)
                .take(1)
                .map(|s| s.to_owned())
                .collect::<String>();
            if !name.is_empty() {
                self.leave(&name);
            }
            return;
        }
        if (self.replying || self.is_privileged(msg))
            && msg
                .message_text
                .to_lowercase()
                .trim()
                .starts_with("!listmods")
        {
            self.client
                .say_in_reply_to(
                    msg,
                    format!(
                        "Moderators: {}",
                        self.moderators
                            .iter()
                            .map(|s| format!("{}, ", s))
                            .collect::<String>()
                    ),
                )
                .await
                .ok();
        }
        if (self.replying || self.is_privileged(msg))
            && msg
                .message_text
                .to_lowercase()
                .trim()
                .starts_with("!listowners")
        {
            self.client
                .say_in_reply_to(
                    msg,
                    format!(
                        "Owners: {}",
                        self.owners
                            .iter()
                            .map(|s| format!("{}, ", s))
                            .collect::<String>()
                    ),
                )
                .await
                .ok();
        }
        if self.is_super_privileged(msg)
            && msg
                .message_text
                .to_lowercase()
                .trim()
                .starts_with("!addmod")
        {
            let name: String = msg
                .message_text
                .split(' ')
                .skip(1)
                .take(1)
                .map(|s| s.to_owned())
                .collect::<String>();
            if !name.is_empty() && self.moderators.insert(name.to_owned()) {
                self.client
                    .say_in_reply_to(msg, format!("Sucessfully added {} to moderators", name))
                    .await
                    .ok();
            } else if !name.is_empty() {
                self.client
                    .say_in_reply_to(msg, format!("{} is already moderator", name))
                    .await
                    .ok();
            }
            return;
        }
        if self.is_super_privileged(msg)
            && msg
                .message_text
                .to_lowercase()
                .trim()
                .starts_with("!remmod")
        {
            let name: String = msg
                .message_text
                .split(' ')
                .skip(1)
                .take(1)
                .map(|s| s.to_owned())
                .collect::<String>();
            if !name.is_empty() && self.moderators.remove(&name) {
                self.client
                    .say_in_reply_to(msg, format!("Sucessfully removed {} from moderators", name))
                    .await
                    .ok();
            } else if !name.is_empty() {
                self.client
                    .say_in_reply_to(msg, format!("{} is already non-moderator", name))
                    .await
                    .ok();
            }
            return;
        }
        if self.is_super_privileged(msg)
            && msg.message_text.to_lowercase().trim().starts_with("!reset")
        {
            self.reset_learning();
            return;
        }
        if self.is_super_privileged(msg)
            && msg
                .message_text
                .to_lowercase()
                .trim()
                .starts_with("!settings")
        {
            self.handle_command_setting(msg).await;
            return;
        }
        if (self.replying || self.is_privileged(msg))
            && msg.message_text.to_lowercase().trim().starts_with("!info")
        {
            self.handle_command_info(msg).await;
            return;
        }
        if (self.replying || self.is_privileged(msg))
            && msg.message_text.to_lowercase().trim().eq("!sglypa")
        {
            self.handle_command_sglypa(msg).await;
            return;
        }
        if self.replying && thread_rng().gen_range(0..A_TILL_PROC) == 0 {
            self.handle_command_sglypa(msg).await;
            return;
        }
        if (self.replying || self.is_privileged(msg))
            && msg.message_text.to_lowercase().trim().eq("!nglypa")
        {
            self.handle_command_nglypa(msg).await;
            return;
        }
        if (self.replying || self.is_privileged(msg))
            && msg.message_text.to_lowercase().trim().starts_with("!")
        {
            self.handle_command_personal_sglypa(msg).await;
            return;
        }
    }

    pub async fn handle_command_stair(&mut self, msg: &PrivmsgMessage, channel: Option<&str>) {
        let tokens: Vec<String> = msg.message_text.split(' ').map(|s| s.to_owned()).collect();
        if let Ok(length) = tokens[1].parse() {
            let message = tokens
                .into_iter()
                .skip(2)
                .map(|s| s + " ")
                .collect::<String>();
            self.say_stair(channel.unwrap_or(&msg.channel_login), length, &message)
                .await;
        }
    }

    pub fn reset_learning(&mut self) {
        unsafe {
            update_markov(self.nmarkov.insert(Markov::new()));
        }
        let _ = self.ntrain_data.insert(Vec::new());
    }

    pub async unsafe fn handle_command_setting(&mut self, msg: &PrivmsgMessage) {
        let tokens: Vec<String> = msg.message_text.split(' ').map(|s| s.to_owned()).collect();
        for i in 1..tokens.len() - 1 {
            match tokens[i].to_uppercase().as_ref() {
                "MIN_REFS" => {
                    if let Ok(val) = tokens[i + 1].parse() {
                        MIN_REFS = val;
                    }
                }
                "MIN_LEN" => {
                    if let Ok(val) = tokens[i + 1].parse() {
                        MIN_LEN = val;
                    }
                }
                "MAX_LEN" => {
                    if let Ok(val) = tokens[i + 1].parse() {
                        MAX_LEN = val;
                    }
                }
                "MIN_SCORE" => {
                    if let Ok(val) = tokens[i + 1].parse() {
                        MIN_SCORE = val;
                    }
                }
                "MAX_TRIES" => {
                    if let Ok(val) = tokens[i + 1].parse() {
                        MAX_TRIES = val;
                    }
                }
                "PROC" => {
                    if let Ok(val) = tokens[i + 1].parse() {
                        A_TILL_PROC = val;
                    }
                }
                "SHOW_REFS" => {
                    if let Ok(val) = tokens[i + 1].parse() {
                        SHOW_REFS = val;
                    }
                }

                _ => {}
            }
        }
    }

    pub async fn handle_command_info(&mut self, msg: &PrivmsgMessage) {
        let tokens: Vec<String> = msg.message_text.split(' ').map(|s| s.to_owned()).collect();
        if tokens.len() > 1 {
            self.client
                .say(
                    msg.channel_login.to_owned(),
                    format!(
                        "@{} Found {} relevant messages from {}",
                        msg.sender.name,
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
            let num = 20;
            let ans = top
                .iter()
                .take(num)
                .map(|(k, v)| format!("{}: {};\t", k, v))
                .collect::<String>();
            self.client
                .say(
                    msg.channel_login.to_owned(),
                    format!("@{} Top {} chatters:\t{}", msg.sender.name, num, ans),
                )
                .await
                .unwrap();
        }
    }

    pub async unsafe fn handle_command_sglypa(&mut self, msg: &PrivmsgMessage) {
        if self.markov.is_none() {
            return;
        }
        if let Ok(result) = self.markov.as_mut().unwrap().generate() {
            println!("{:?}", result);
            if !SHOW_REFS {
                self.client
                    .say_in_reply_to(msg, format!("Sglypa: {}", &result.text.to_owned()))
                    .await
                    .ok();
            } else {
                self.client
                    .say_in_reply_to(
                        msg,
                        format!(
                            "Sglypa: {} [{}]",
                            &result.text.to_owned(),
                            result
                                .refs
                                .iter()
                                .map(|&id| format!(
                                    "{},",
                                    self.train_data
                                        .as_ref()
                                        .unwrap()
                                        .get(id)
                                        .unwrap()
                                        .meta
                                        .as_ref()
                                        .unwrap()
                                        .to_owned()
                                ))
                                .collect::<String>()
                        ),
                    )
                    .await
                    .ok();
            }
        }
    }

    pub async unsafe fn handle_command_nglypa(&mut self, msg: &PrivmsgMessage) {
        if self.nmarkov.is_none() {
            return;
        }
        if let Ok(result) = self.nmarkov.as_mut().unwrap().generate() {
            println!("{:?}", result);
            if !SHOW_REFS {
                self.client
                    .say_in_reply_to(msg, format!("Sglypa: {}", &result.text.to_owned()))
                    .await
                    .ok();
            } else {
                self.client
                    .say_in_reply_to(
                        msg,
                        format!(
                            "Sglypa: {} [{}]",
                            &result.text.to_owned(),
                            result
                                .refs
                                .iter()
                                .map(|&id| format!(
                                    "{},",
                                    self.ntrain_data
                                        .as_ref()
                                        .unwrap()
                                        .get(id)
                                        .unwrap()
                                        .meta
                                        .as_ref()
                                        .unwrap()
                                        .to_owned()
                                ))
                                .collect::<String>()
                        ),
                    )
                    .await
                    .ok();
            }
        }
    }

    pub async unsafe fn handle_command_personal_sglypa(&mut self, msg: &PrivmsgMessage) {
        if self.personal_markov.is_none() {
            return;
        }
        if let Some(name) = msg.message_text.to_lowercase().strip_prefix("!") {
            if self.personal_markov.is_some()
                && self.personal_markov.as_mut().unwrap().contains_key(name)
            {
                if let Ok(result) = self
                    .personal_markov
                    .as_mut()
                    .unwrap()
                    .get(name)
                    .unwrap()
                    .generate()
                {
                    println!("{:?}", result);
                    self.client
                        .say_in_reply_to(msg, format!("Sglypa: {}", &result.text.to_owned()))
                        .await
                        .ok();
                }
            }
        }
    }

    pub fn handle_learn(&mut self, msg: &PrivmsgMessage) {
        if !learn_filter(msg.sender.name.to_owned(), msg.message_text.to_owned()) {
            return;
        }
        let name = &msg.sender.login;
        if self.markov.is_some() && self.train_data.is_some() {
            self.train_data.as_mut().unwrap().push(InputData {
                text: msg.message_text.to_owned(),
                meta: Some(name.to_owned()),
            });
            self.markov.as_mut().unwrap().add_to_corpus(vec![self
                .train_data
                .as_mut()
                .unwrap()
                .last()
                .unwrap()
                .clone()]);
        }
        if self.personal_markov.is_some() && self.personal_train_data.is_some() {
            if let Some(train_data) = self.personal_train_data.as_mut().unwrap().get_mut(name) {
                train_data.push(InputData {
                    text: msg.message_text.to_owned(),
                    meta: Some(name.to_owned()),
                });
            }
            if let Some(markov) = self.personal_markov.as_mut().unwrap().get_mut(name) {
                markov.add_to_corpus(vec![self
                    .train_data
                    .as_mut()
                    .unwrap()
                    .last()
                    .unwrap()
                    .clone()]);
            }
        }
        if self.nmarkov.is_some() && self.ntrain_data.is_some() {
            self.ntrain_data.as_mut().unwrap().push(InputData {
                text: msg.message_text.to_owned(),
                meta: Some(name.to_owned()),
            });
            self.nmarkov.as_mut().unwrap().add_to_corpus(vec![self
                .train_data
                .as_mut()
                .unwrap()
                .last()
                .unwrap()
                .clone()]);
        }
    }
}

fn learn_filter(name: String, body: String) -> bool {
    !body.contains("Tier")
        && !body.to_lowercase().contains("sglypa")
        && !body.starts_with("!")
        && body.split(" ").count() >= STATE_SIZE
        && !body.to_lowercase().contains("-.-.-")
        && !body.chars().all(|c| c.is_ascii())
    // && (!name.to_lowercase().eq("kabachoke") || (!body.to_lowercase().contains(":confident:") && !body.to_lowercase().contains("мы зависли")))
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
    sglypa.train_from_vods(
        true,
        Some(|p: &PathBuf| {
            // !p.components()
            //     .map(|c| c.as_os_str().to_str())
            //     .any(|c| c.map_or(false, |c| c.starts_with("marathon")))
            p.components()
                .map(|c| c.as_os_str())
                .any(|c| c.eq("red_pondaa")) // || c.eq("toopenya"))
        }),
        Some(learn_filter),
    );
    sglypa.join("gosuto_botto");
    sglypa.reset_learning();
    // sglypa.join("red_pondaa");
    sglypa.run().await;
}
