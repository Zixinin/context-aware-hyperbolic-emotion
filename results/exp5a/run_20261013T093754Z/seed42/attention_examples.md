# Exp5A label-wise attention sanity examples

These are diagnostics, not evaluation metrics.

## Example 1

- ID: `ed7pe2z`
- Text: Haha, my apologies!
- True labels: amusement
- Predicted labels at 0.30: amusement, remorse

- amusement: `Ha` (0.11173), `##ha` (0.11166), `[CLS]` (0.11121), `##pologies` (0.11115), `a` (0.11105)
- anger: `##ha` (0.11130), `Ha` (0.11129), `my` (0.11125), `a` (0.11122), `,` (0.11121)
- fear: `my` (0.11126), `,` (0.11125), `##ha` (0.11122), `Ha` (0.11117), `a` (0.11114)
- joy: `##ha` (0.11190), `Ha` (0.11175), `[SEP]` (0.11155), `##pologies` (0.11104), `[CLS]` (0.11088)
- remorse: `my` (0.11159), `,` (0.11157), `##pologies` (0.11146), `a` (0.11143), `!` (0.11129)

## Example 2

- ID: `eex0tm9`
- Text: It surprises me that he's a mod some days...
- True labels: surprise
- Predicted labels at 0.30: surprise

- anger: `a` (0.05903), `days` (0.05896), `'` (0.05893), `some` (0.05891), `s` (0.05890)
- fear: `[CLS]` (0.05893), `.` (0.05892), `he` (0.05891), `that` (0.05890), `.` (0.05888)
- joy: `'` (0.05903), `some` (0.05900), `that` (0.05898), `[CLS]` (0.05893), `he` (0.05889)
- surprise: `'` (0.05906), `that` (0.05901), `some` (0.05900), `s` (0.05897), `[CLS]` (0.05896)

## Example 3

- ID: `ee1vrgl`
- Text: Seems to be fake. Checked the Morning Mix website, no articles from Nov 13^(th) 2018 that match the title shown.
- True labels: neutral
- Predicted labels at 0.30: neutral

- anger: `th` (0.03358), `articles` (0.03357), `that` (0.03356), `be` (0.03354), `13` (0.03353)
- fear: `Seems` (0.03356), `fake` (0.03352), `match` (0.03352), `to` (0.03346), `,` (0.03345)
- joy: `the` (0.03362), `Check` (0.03361), `website` (0.03361), `from` (0.03350), `##ed` (0.03347)
- neutral: `that` (0.03348), `13` (0.03347), `)` (0.03345), `,` (0.03344), `the` (0.03344)

## Example 4

- ID: `edw23ri`
- Text: I have faith
- True labels: neutral
- Predicted labels at 0.30: optimism, neutral

- anger: `I` (0.20068), `[CLS]` (0.20001), `faith` (0.19993), `have` (0.19992), `[SEP]` (0.19946)
- fear: `[CLS]` (0.20045), `I` (0.20024), `faith` (0.20017), `have` (0.19959), `[SEP]` (0.19955)
- joy: `[SEP]` (0.20042), `faith` (0.20036), `have` (0.19998), `I` (0.19970), `[CLS]` (0.19953)
- optimism: `I` (0.20107), `[CLS]` (0.20044), `have` (0.20037), `faith` (0.19978), `[SEP]` (0.19834)
- neutral: `[CLS]` (0.20113), `have` (0.20057), `faith` (0.20008), `I` (0.19935), `[SEP]` (0.19887)

## Example 5

- ID: `eddf4lj`
- Text: Oh. :[
- True labels: neutral
- Predicted labels at 0.30: neutral

- anger: `.` (0.16720), `[CLS]` (0.16693), `:` (0.16689), `[` (0.16682), `Oh` (0.16621)
- fear: `[` (0.16681), `:` (0.16681), `.` (0.16679), `[CLS]` (0.16673), `Oh` (0.16667)
- joy: `Oh` (0.16727), `[SEP]` (0.16677), `[` (0.16664), `[CLS]` (0.16652), `.` (0.16645)
- neutral: `[CLS]` (0.16723), `[` (0.16698), `.` (0.16683), `Oh` (0.16680), `:` (0.16657)
