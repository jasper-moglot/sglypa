use markov_strings::*;
use twitch_irc::login::{CredentialsPair, StaticLoginCredentials};
use twitch_irc::message::ServerMessage;
use twitch_irc::{ClientConfig, SecureTCPTransport, TwitchIRCClient};

const need_train: bool = true;

#[tokio::main]
pub async fn main() {
    let twitch_name = "gosuto_botto".to_owned();
    let twitch_token = "oauth:r7sgsai3jqws965ihkiujw6wpbld6s"
        .to_owned()
        .replacen("oauth:", "", 1);
    let channel_to_join = "gosuto_botto".to_owned();
    let config = ClientConfig {
        login_credentials: StaticLoginCredentials {
            credentials: CredentialsPair {
                login: twitch_name.clone(),
                token: Some(twitch_token),
            },
        },
        ..ClientConfig::default()
    };

    let (mut incoming_messages, client) =
        TwitchIRCClient::<SecureTCPTransport, StaticLoginCredentials>::new(config);

    // Instantiate the generator
    let mut markov = Markov::new();
    // Optional: specify a state size
    markov.set_state_size(2).unwrap(); // Default: 2
                                       // Define a results filter
    markov
        .set_filter(|r| {
            // A minimal relative score and number of references
            // The thresholds are relative to your input
            r.score > 5 && r.refs.len() > 2
            // We want to generate random tweets
            && r.text.len() <= 280
            // No commands
            && !r.text.starts_with("!")
            // No mentions
            && !r.text.contains("@")
            // No urls
            && !r.text.contains("http")
        })
        .set_max_tries(100);

    if need_train {}

    client.join(channel_to_join.clone()).unwrap();
    let join_handle = tokio::spawn(async move {
        while let Some(message) = incoming_messages.recv().await {
            match message {
                ServerMessage::Privmsg(msg) => {
                    // `markov.add_to_corpus(vec![InputData {
                    // `    text: msg.message_text.to_owned(),
                    // `    meta: None,
                    // `}]);
                    if let Ok(result) = markov.generate() {
                        // client
                        //     .say(channel_to_join.clone(), result.text.to_owned())
                        //     .await
                        //     .unwrap();
                        println!("generated: {}", result.text);
                    }
                    println!(
                        "(#{}) {}: {}",
                        msg.channel_login, msg.sender.name, msg.message_text
                    );
                }
                ServerMessage::Whisper(msg) => {
                    println!("(w) {}: {}", msg.sender.name, msg.message_text);
                }
                _ => {}
            }
        }
    });
    // keep the tokio executor alive.
    // If you return instead of waiting the background task will exit.
    join_handle.await.unwrap();
}
