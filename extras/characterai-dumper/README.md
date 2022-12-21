# CharacterAI Dumper Userscript

This userscript allows you to download your saved messages with any bot you've ever talked to, given you can reach their chat history page.

## How to use

- Install a userscript manager.
    - Personally I tested this with [Violentmonkey](https://violentmonkey.github.io/get-it/) on Firefox, but I think Greasemonkey and Tampermonkey should work as well.
- [Click here](TODO) and install the userscript.
- Now, while you're talking to a character, click on "View Saved Chats" to go to the histories page:
    - TODO: Image here.
- After a few seconds, a `Download` link should pop up next to the "Your past conversations with so-and-so" header:
  - TODO: Image here.
- Clicking on the link will download a `.json` file containing the bot's basic info (name, description, greeting) and all the interactions you've ever had with it.

**NOTE:** The script attempts to anonymize the dumped data (it scrubs known sensitive fields and attempts to replace any instances of your name within messages), but if you're paranoid, you should open the downloaded JSON and search for your username/email/display name just to make sure.

## Troubleshooting

If the `Download` link doesn't show up after a few seconds and you're on the proper page, check the DevTools console for errors.

The most probable causes of breakage are:
- You have some browser extension which stopped the userscript from loading its external dependencies properly; or
- CharacterAI changed their API around in ways the script didn't expect, and I'll need to release an update.
