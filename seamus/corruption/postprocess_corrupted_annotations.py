import click
import json
import re
import spacy_alignments as tokenizations

JSON_MARKDOWN_REGEX = re.compile(r".*?```json(.*?)```", re.DOTALL)
JSON_REGEX = re.compile(r".*?(\{.*\})", re.DOTALL)

from seamus.constants import SPLIT_TO_PATH

from sacremoses import MosesDetokenizer
from spacy.lang.en import English
from tqdm import tqdm

nlp = English()

DETOKENIZER = MosesDetokenizer(lang="en")

MANUAL_ARGUMENT_MAPPINGS = {
    "train": {
        "The operation of a motor vehicle in a manner that endangers or is likely to endanger persons or property.": "[t] he operation of a motor vehicle in a manner that endangers or is likely to endanger persons or property",
        "the 1980s Finals": "their 1980s Finals matches",
        "first - team in vault in the first National Association of Collegiate Gymnastics Coaches (NACGC)": "second - team in vault in the first National Association of Collegiate Gymnastics Coaches (NACGC)",
        "Ayami Kojima's drawings": "Kojima's drawings",
        "26 June 1998 Trec Smith": "26 June 1998",
        "Lou Gehrig": "Gehrig",
        "'Westlife have claimed that 'Us Against the World' is the best track on their recent Back Home album": "Westlife have claimed that 'Us Against the World' is the best track on their recent Back Home album",
        "from the field": "from the stands",
        "two Egyptian MiG- 19s": "two Egyptian MiG - 19s",
        "the back of the rear case hidden behind the foot.": 'the back of the rear case hidden behind the" foot',
        "the approximately 300 delegates to the first General Council represented a variety of independent churches and networks of churches.": "The approximately 300 delegates to the first General Council represented a variety of independent churches and networks of churches",
        "I think this [past] year we stopped the momentum.": "I think this [past] year we stopped the momentum",
        "England's players who seemed repentant for their earlier failings.": "England's players who seemed repentant for their earlier failings",
        "the boy's realised the error of his ways: \"I've been doing what I like when I like how I like â€“ it's joyless.\"": "the boy's realised the error of his ways",
        'the error Ca n\'t Convert "2201010001" to long.': 'an error Ca n\'t Convert "2201010001" to long',
        "the blame game has many pointing fingers at the plant operator and at the national oversight agency.": "the blame game has many pointing fingers at the plant operator and at the national oversight agency",
        "the British airliner": "another British airliner",
        "I've been surprised at how few people are willing to get annoyed with me over it.": "I've been surprised at how few people are willing to get annoyed with me over it",
        "when the Croswell is slated to begin": "when the construction is slated to begin",
        "People have been calling the office all week telling us there are churches still meeting.": "People have been calling the office all week telling us there are churches still meeting",
        "a thought-provoking anticlimax is... well, still an anticlimax.": "A thoughtful anticlimax is... well, still an anticlimax",
        "the night of February": "the last week of February",
        "the so-called 'Ordsall Chord'": "The so- called 'Ordsall Chord'",
        "a grandiose spectacle is further proof of the greatness... of a unique civilization that extends into the depths of history": "This grandiose spectacle is further proof of the greatness... of a unique civilization that extends into the depths of history",
        "the election of new President commences": "election of new President commences",
        "that “went like a dream ” but ended in disaster when he later realized he had failed to turn on his tape recorder": "that “went like a drea m ” but ended in disaster when he later realized he had failed to turn on his tape recorder",
        "the signature is a signature of destruction": "our signature is a signature of destruction",
        "To step away from international cricket after 16 years is something I will miss dearly - I will miss the guys probably more than anything, but pulling on the shirt and playing for Ireland was the only thing I ever dreamed of growing up. To have played for Ireland 292 times is something I am very proud of - if I had thought I'd have made it 10 times growing up I would have snapped your hand off, so 292 times is something I'm very proud of.": "To step away from international cricket after 16 years is something I will miss dearly",
        "The affiliation will integrate the clinical enterprises, bringing the strength of Tuality's deep community relationships together with OHSU's distinct role as a provider of highly complex and advanced specialty care.": "The affiliation will integrate the clinical enterprises",
        "because of his “financial means and international connections to flee and remain at large.”": "because of his “financial means and international connections to flee and remain at large. ”",
        "the President": "Professor Alice P. Gast",
        "an hour and 45 - minute set": "the entire hour and 45 - minute set",
        "he was 'plainly a terrorist'": 'al - Harith was "plainly a terrorist"',
        "a measure that would require girls 18 or under to notify their parents before getting an abortion": "an initiative that would require girls 18 or under to notify their parents before getting an abortion",
        "the two young lives were wasted in what was a criminal enterprise": "these two young lives were wasted in what was a criminal enterprise",
        "the small, run - down Apostolic Faith Mission at 312 Azusa Street in Los Angeles, California": "the small, run -down Apostolic Faith Mission at 312 Azusa Street in Los Angeles, California",
        "on Christmas Eve 2007": "Christmas Eve",
        "the village of Hadareni": "Hadareni",
        "the site of John Walker, the editor of PC gaming site, Rock Paper Shotgun": "the personal site of John Walker",
        "It's a one - stop shop; all communication tools are on one platform.": "It's a one - stop shop; all communication tools are on one platform",
        "[b-hive] will definitely spark some customers to look at what else they can do and what else they can offer their end customers.": "[b-hive] will definitely spark some customers to look at what else they can do and what else they can offer their end customers",
        "a special variety of white bread called “milk bread” must contain 6% milk by Provincial decree!": "a special variety of white bread called �milk bread� must contain 6% milk by Provincial decree!",
        "Dreikönigsabend is always a special event.": '"Dreikönigsabend" is always a special event',
        'NBC\'s "Meet the Press"': 'NBC\'s "Meet the Press,"',
        "creates gullies": "creating gullies",
        "It could be that those rare manzanitas, they have seeds everywhere.": "It could be that those rare manzanitas, they have seeds everywhere",
        "the temple bell": "temple bells",
        "criminals": "criminal suspects",
        "the plaintiff owed to the plaintiff a duty to manufacture an automobile with which it was safe to collide.": "the defendant owed to the plaintiff a duty to manufacture an automobile with which it was safe to collide",
        "the Bendigo Three": "The so- called 'Bendigo Three'",
        "who have been abused by both sides": "Children on both sides",
        "from 8 pm to 8 am from Monday to Friday": "from 7 pm to 8 am from Monday to Friday",
        'Aetna, a major US health care company (aetna.com) advises: "The health card entitles you to free treatment at public hospitals and subsidised prices on prescription medicines."': "The health card entitles you to free treatment at public hospitals and subsidised prices on prescription medicines",
        "a day after Defence Secretary Robert Gates told US troops the Iraq mission was in its 'endgame'.": 'a day after Defence Secretary Robert Gates told US troops the Iraq mission was in its "endgame"',
        "the Department 0.` Information propaganda fund": "Department 0.` Information propaganda fund",
        "a folded ear": "folded ears",
        "what areas of spiritual growth they especially need to ask God's help in dealing with.": "what amendments of life or areas of spiritual growth they especially need to ask God's help in dealing with",
        "he may have fallen against a glass partition separating his home's kitchen from the garden": "Brunkert may have fallen against a glass partition separating his home's kitchen from the garden",
        "blind spots that prevent the formation of ": "blind spots",
        "a bowl": "this classic restaurant potato dish",
        "the city centre population of Venezia and Chioggia and inhabitants of the islands of the lagoon": "city centre population of Venezia and Chioggia and inhabitants of the islands of the lagoon",
        "historic amounts of rain": "record - breaking rain",
        "the Belgian parachutists": "Belgian parachutists",
        "James “Jim” Arthur Brown": "James “Ji m ” Arthur Brown",
        "November 9 to November 13": "Nov. 9 to Nov. 13",
        "calm and distance.": "genteel but distant",
        "putting in our footings": "put in our footings",
        "during the attacks.": "over a 6 - month period",
        "the late - night regulars in a Hamburg bar in the 1960s": "late - night regulars in a Hamburg bar in the 1960s",
        "more than 1.2 million effects": "distortion effects",
        "a blackmail attempt": "blackmail attempt",
        "the cabinet resolved not to renew or issue new gold mining licences": "the cabinet's unexpected decision not to renew the mining licence",
        "translated radical Arabic books and videos into English for the website At Tibyan": "translating radical Arabic books and videos into English for the website At Tibyan",
        "The Middle East is in dire need of broader - albeit imaginative - regional integration.": "The Middle East, on the other hand, is in dire need of broader - albeit imaginative - regional integration",
        "the ideal mothering breed.": "ideal mothering breed",
        "It’s going to improve relations between NOPD and immigrant communities, and it's going to make all our communities safer by making police more accountable.": "It's going to improve relations between NOPD and immigrant communities, and it's going to make all our communities safer by making police more accountable",
        "the debris of historical trauma": "this debris of historical trauma",
        "at the annual meeting of the International Studies Association in San Francisco": "at the annual meeting of the International Studies Association",
        "in their new home, they will occupy 'slightly upgraded cases'": 'In their new home, they will occupy "slightly upgraded cases"',
        "a rifle": "his rifle",
        "Jantzen Beach was heralded as Portland's Million Dollar Playground.": "Jantzen Beach Amusement Park was heralded as Portland's Million Dollar Playground",
        "hurled towards the other end": "hurtled towards the other end",
        "having no cuts at all": "have no cuts at all",
        "the men wanted to kill some Iraqis": 'they "wanted to kill some Iraqis"',
        "the girl was raped and murdered.": "The men murdered her and her family",
        "to Texas": "in Texas",
        "when I reached the top": "when you reached the top",
        "skimming off and discarding the residue that forms on top of water": "Skim off and discard the residue that forms on top of water",
        "the official “Keeper of the Great Clock of Wells”": "the official “Keeper of the Great Clock of Wells ”",
        "the borough": "their borough",
        "Discovery’s crew": "Discovery crew",
        "captured hypervelocity orbital debris and natural micrometeoroid particles": "capture hypervelocity orbital debris and natural micrometeoroid particles",
        "the mission is to connect water professionals; enrich the expertise of water professionals; increase the awareness of the impact and value of water; and provide a platform for water sector innovation.": "our mission is to connect water professionals; enrich the expertise of water professionals; increase the awareness of the impact and value of water; and provide a platform for water sector innovation",
        "the creator of Katie Morag": "her creator",
        "the concert hall": "the same concert hall",
        "the Sacramento hotel": "a Sacramento hotel",
        "Display our heritage helps the community cherish Ottawa's neighbourhoods, communities and history.": "Displaying our heritage helps the community cherish Ottawa's neighbourhoods, communities and history",
        "the Jaguars punter": "The Jacksonville Jaguars punter",
        "the 50th anniversary of the Marine Offences Act which ended pirate radio broadcasts": "50th anniversary of the Marine Offences Act which ended pirate radio broadcasts",
        "that he must make a one - time only switch with FIFA if he wants to to play for the US in an official event": "he must make a one - time only switch with FIFA if he wants to to play for the US in an official event",
        "coached under Lyon, Clarko and Mick": "apprenticeship in footy under Lyon, Clarko and Mick",
        "shed her smooth and pretty image completely": "sheds her smooth and pretty image completely",
        "He turns 80 on Feb. 1": "Lantos, who turns 80 on Feb. 1",
        "Feb. 22, 2006": "2/22/2006",
        "the old Astra 2A north beam, even larger 2.4 m dishes could struggle to receive the channels 24/7, especially mid afternoon.": "When channels were on the Astra 2A north beam, even larger 2.4 m dishes could struggle to receive the channels 24/7, especially mid afternoon",
        "rocks tumbling into the sea and causing sea levels to rise": "Rocks tumbling into ocean causing sea level rise",
        "the night sky": "the predawn sky",
        "he was quick and accurate in his judgment of length and speed": "Bradman was quick and accurate in his judgment of length and speed",
        "February 28": "Feb. 28",
        "the Isle of Skye estate.": "his Isle of Skye estate",
        "the French tricolour, to signify the victory over the French at Waterloo": "a French tricolour, to signify the victory over the French at Waterloo",
        'the mummies will occupy "slightly upgraded cases"': 'they will occupy "slightly upgraded cases"',
        "Thursday, March 22, 2012": "Thu., March 22, 2012",
        "the Ukrainian Autocephalous Orthodox church (UAOC)": "the Ukrainian Autocephalous Orthodox church",
        "the top of the world!": "top of the world",
        "thousands of slightly wounded men": "hundreds of slightly wounded men",
        "in Washington": "Washington",
        "France": "French",
        "the impeachment trial": "the impeachment proceedings",
        "providing accessibility is conditioned on whether providing access through a fixed lift is “readily achievable.”": "If accessibility is not readily achievable, businesses should develop plans for providing access into the pool when it becomes readily achievable in the future",
        "a commissioner with a “managing director” role to lead the oversight of overall service and governance improvement, driving performance": "a commissioner with a “managing director ” role to lead the oversight of overall service and governance improvement, driving performance",
        "the Royal College of Nursing (RCN) Scotland": "Royal College of Nursing (RCN) Scotland",
        "California's Remington Park": "Remington Park",
        "until she will make Edward wait only as far as her high - school graduation": "only as far as her high - school graduation",
        "A woman who identified herself to KTVT - TV as Wetteland's wife told the station that he was '100 percent innocent.'": 'A woman who identified herself to KTVT - TV as Wetteland\'s wife told the station that he was "100 percent innocent."',
        "in 1975": "in May 1975",
        "a stupid act on its own and a stupid precedent to set": 'a "stupid act on its own and a stupid precedent to set"',
        "the “Black Sheep”": "The famed “Black Sheep ”",
        "the famed “Black Sheep” patch, name, and colors will be maintained by Air Force historians for reactivation when needed.": "The famed “Black Sheep ” ” patch, name, and colors will be maintained by Air Force historians for reactivation when needed",
    },
    "dev": {},
    "test": {
        "A pretty entertaining game": 'a "pretty entertaining game',
        "the U.S.": "U.S.",
        "The first palestinian suicide attack": "The first Palestinian suicide attack",
        "To injure his throat": "injuring his throat",
        "Using expeller pressing": "expeller pressing",
        "September 6, 2009": "September 5, 2009",
        "Construction of hard surface road": "construction of of hard surface road",
        "A tour": "the tour",
        "The fa": "The FA",
        "The supreme court": "The Supreme Court",
        "During the final days of the fifth phase campaign": "During the final days of the Fifth Phase Campaign",
        "The use of the bios": "The use of the BIOS",
        "The lewis bronzeville five": "the Lewis Bronzeville Five",
        "On 21 october 2016": "On 21 October 2016",
        "The so-called herzog commission": "The so-called Herzog Commission",
        "Germany": "German",
        "The phased reopening plan": "their phased reopening plan",
        "The cryptic region": "the 'cryptic region'",
        "A spokesperson for burnham": "A spokesperson for Burnham",
        "Kristiansen had been paid £10,000 by richardson to start the fire.": "Kristiansen, a former SAS soldier, had been paid £10,000 by Richardson to start the fire",
        'Brought up during a segment, "do not play", on the tonight show starring jimmy fallon': 'in a segment, "Do not Play"',
        "On double lp": "On Double LP",
        "The image of st. andrew": "The image of St. Andrew",
        "The clp": "The CLP",
        "After cameron's firing": "After Cameron's firing",
        'The book of documents "kim busik"': 'the Book of Documents "Kim Busik."',
        "Then": "wherever he went",
        "The transition area between the scarps and the permafrost": "This transition area between the scarps and the permafrost",
        "The 29-year-old writer / producer": "The 29- year-old writer / producer",
        "1.8 million pounds (Rs 18 crore)": "1. 8 million pounds (Rs 18 crore)",
        "the process of forecasting the path of the economy": "process of forecasting the path of the economy",
        "green fuel": 'so - called "green" fuel',
        "October 2013": "December 31, 2013",
        "a ball of rice": "the ball of rice",
        "a remarkable geological phenomenon": "this remarkable geological phenomenon",
        "the clubs' union the LEGA": "LEGA",
        "3 tablespoons olive oil": "3 tbsp olive oil",
        "up to Bhaktapur": "Bhaktapur",
        "to the top eastern part of Nepal": "in the top eastern part of Nepal",
        "to the fulfillment of our manifest destiny": "the fulfillment of our manifest destiny",
        "to the stewards": "the stewards",
        "the road from Eleusis to Thebes": "the direct road from Eleusis to Thebes",
        "to Pathivara temple": "a temple of goddess Pathivara",
        "non-cancerous changes": "non - cancerous changes",
        "steamed with glutinous rice": "made with glutinous rice",
        "a show that almost didn't get renewed this season": "a show that almost did n't get renewed this season",
        "a 'saltire' (or diagonal cross)": "the 'saltire' (or diagonal cross)",
        "confectionary sunflower seeds": '"confectionary" sunflower seeds',
        "A welded frame offers the necessary support for pressing with a 3-ton hydraulic jack.": "A welded frame offers the necessary support for pressing with a 3 - ton hydraulic jack",
        "the emperor's absence": "emperor's absence",
        "her personal experience in prison": "Henry's personal experience in prison",
        "a perfect, insect-made bore, except that it is too narrow for what I wanted, so I gouged it out wider": "a perfect, insect - made bore, except that it is too narrow for what I wanted, so I gouged it out wider",
        "hundreds of African migrants": "hundreds if not thousands of African migrants",
        "Ayub was supposedly shocked when he heard that some of his 'children' had called him a dog.": "Ayub was supposedly shocked when he heard that some of his' children '(the term he used to describe his subjects), had called him a dog",
    },
}


@click.command()
@click.argument("corrupted_templates_file", type=str)
@click.argument("output_file", type=str)
@click.option(
    "--original-file",
    type=str,
    default=SPLIT_TO_PATH["test"],
    help="The file that contains the original templates that were corrupted",
)
@click.option("--doc", "-d", type=click.Choice(["source", "report"]), default="report")
@click.option("--split", "-s", type=str, default="test")
def extract_corrupted_templates(
    corrupted_templates_file, output_file, original_file, doc, split
):
    with open(original_file, "r") as f:
        data = json.load(f)

    with open(corrupted_templates_file, "r") as f:
        corrupted_templates_raw = [json.loads(line) for line in f]
        corrupted_templates_raw = {
            ex["instance_id"]: ex["response"] for ex in corrupted_templates_raw
        }

    out = []
    for ex in tqdm(data, "Processing templates..."):
        instance_id = ex["instance_id"]
        if instance_id not in corrupted_templates_raw:
            out.append(ex)
            continue

        corrupted_template_raw = corrupted_templates_raw[instance_id]
        regex_match = re.search(JSON_REGEX, corrupted_template_raw)
        if not regex_match:
            print(f"Could not find JSON in response for example {instance_id}")
            out.append(ex)
            continue
        else:
            try:
                corrupted_template = json.loads(regex_match.group(1).strip())
            except json.JSONDecodeError:
                print(f"Could not parse JSON in response for example {instance_id}")
                out.append(ex)
                continue

        template_key = "report_template" if doc == "report" else "source_template"
        text_tok = ex[doc]
        text_detok = DETOKENIZER.detokenize(text_tok)
        text_detok_lower = text_detok.lower()
        char2tok, tok2char = tokenizations.get_alignments(list(text_detok), text_tok)
        orig_template = ex[template_key]

        # ad-hoc fix
        if instance_id == "EN-0548-981-frame-Intentional_traversing":
            corrupted_template["Goal"] = {"arguments": ["into the Venetian lagoon"]}

        # The original template and the corrupted
        # template should have all the same roles
        assert (
            orig_template.keys() == corrupted_template.keys()
        ), f"Role mismatch for example {instance_id}"

        # Process the corrupted annotations
        new_template = {}
        for role, role_data in orig_template.items():
            orig_args = [tuple(arg["tokens"]) for arg in role_data["arguments"]]
            corrupted_args = [
                a for a in corrupted_template[role]["arguments"] if a != ""
            ]
            new_template[role] = {"arguments": []}
            for arg in corrupted_args:
                arg_toks = tuple([tok.text for tok in nlp(arg)])
                try:
                    # if this argument exists in the original template,
                    # add it to the new template
                    matched_arg_idx = orig_args.index(arg_toks)
                    new_template[role]["arguments"].append(
                        role_data["arguments"][matched_arg_idx]
                    )
                except ValueError:
                    # this argument does not exist in the original template;
                    # we need to add it. We heuristically locate the first
                    # occurrence in the document and use this as the new
                    # argument
                    orig_arg = arg
                    if arg in MANUAL_ARGUMENT_MAPPINGS[split]:
                        # some corrupted arguments failed to be copied EXACTLY
                        # from the text; these must be fixed manually
                        arg = MANUAL_ARGUMENT_MAPPINGS[split][arg]

                    # try stripping punctuation
                    elif arg[-1] in {".", "!", "?", ","}:
                        arg = arg[:-1]

                    arg_lower = arg.lower().strip()
                    assert (
                        arg_lower in text_detok_lower
                    ), f"Could not find argument '{orig_arg}' in text:\n\n{text_detok}"
                    arg_start_char = text_detok_lower.index(arg_lower)
                    arg_end_char = arg_start_char + len(arg_lower)
                    arg_start_tok = char2tok[arg_start_char][0]
                    arg_end_tok = char2tok[arg_end_char - 1][-1]
                    new_template[role]["arguments"].append(
                        {
                            "tokens": text_tok[arg_start_tok : arg_end_tok + 1],
                            "start": arg_start_tok,
                            "end": arg_end_tok,  # inclusive
                        }
                    )

        # replace the original template with the new one
        ex[template_key] = new_template
        out.append(ex)

    with open(output_file, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    extract_corrupted_templates()
