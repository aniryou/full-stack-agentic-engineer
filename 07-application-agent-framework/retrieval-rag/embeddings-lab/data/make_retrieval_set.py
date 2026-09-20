"""A small retrieval corpus with planted phenomena for notebook 05:
  * paraphrase query where lexical search misses (needs a synonym bridge doc)
  * exact-identifier query (E-4417 vs E-4471) where dense retrieval confuses codes
  * negation query that both retrievers get wrong
  * ordinary topical queries so aggregate metrics are meaningful
36 docs across 3 domains, 10 judged queries."""
import json

D = []
def doc(i, title, text): D.append({"id": i, "title": title, "text": text})

# --- coffee / espresso support ---------------------------------------------
doc("c01", "Dialing in grind size", "Adjust the grinder finer if the shot runs fast and sour. Coarser grind slows extraction and reduces bitterness in espresso.")
doc("c02", "Water temperature for brewing", "Brew coffee between 92 and 96 celsius. Water that is too hot scalds the grounds and water that is too cool under extracts.")
doc("c03", "Removing mineral buildup", "Hard water leaves limescale inside the brewer boiler and the brew unit. Flush the system monthly with a citric solution to dissolve the mineral buildup and restore flow.")
doc("c04", "Steam wand care", "Purge the steam wand after every use and wipe it with a damp cloth so milk residue does not clog the tip.")
doc("c05", "Descaling guide", "To descale the espresso machine, run the descale program with descaling solution. Descaling removes limescale and mineral buildup from the boiler.")
doc("c06", "Storing coffee beans", "Keep beans in an airtight opaque container away from heat and light. Do not refrigerate beans because moisture ruins the aroma.")
doc("c07", "Error E-4417 water sensor", "The display shows error code E-4417 when the machine stops. Reseat the water tank firmly and restart the machine to clear the error code.")
doc("c08", "Error E-4471 pump fault", "The display shows error code E-4471 when the machine stops. Power cycle the pump unit and restart the machine to clear the error code.")
doc("c09", "Milk frothing basics", "Stretch the milk with the wand tip near the surface, then submerge to spin a whirlpool until the jug is hot to the touch.")
doc("c10", "Portafilter cleaning", "Soak the portafilter and basket in cleaning powder weekly. Coffee oils build up and turn shots rancid if the basket is dirty.")
doc("c11", "Choosing a burr grinder", "Flat burrs give clarity while conical burrs give body. For espresso a stepless burr grinder allows fine adjustment.")
doc("c12", "Pre infusion explained", "Pre infusion wets the puck at low pressure before full pressure, reducing channeling and evening out extraction.")

# --- hiking / travel --------------------------------------------------------
doc("h01", "Steep ridge climb", "The ridge trail is steep and exposed, gaining 900 meters over rocky switchbacks. Only experienced hikers should attempt this steep route in wet weather.")
doc("h02", "Gentle valley loop", "The valley loop is a flat easy walk beside the river, suitable for families. The gentle path has shade, benches, and no significant climbing.")
doc("h03", "Day hike packing list", "Carry water, snacks, a rain shell, sunscreen, a map, and a headlamp. Pack layers because mountain weather changes quickly on a day hike.")
doc("h04", "Altitude sickness symptoms", "Above 2500 meters watch for headache, nausea, dizziness, and poor sleep. Altitude sickness improves by descending and resting.")
doc("h05", "Leave no trace", "Pack out all rubbish, stay on marked trails, and keep distance from wildlife so the trail stays wild for others.")
doc("h06", "Waterfall boardwalk", "A short boardwalk leads to the waterfall viewpoint. The path is flat and wheelchair accessible with railings the whole way.")
doc("h07", "Trail running shoes", "Trail shoes need aggressive lugs for grip on mud and a rock plate to protect the foot on technical ground.")
doc("h08", "River crossing safety", "Unbuckle your pack, face upstream, and use poles. Never cross a river above knee depth after heavy rain.")
doc("h09", "Sunrise summit tips", "Start hiking two hours before dawn with a headlamp. The summit is cold before sunrise so bring an insulated jacket.")
doc("h10", "Camping permits", "Overnight camping in the reserve requires a permit booked online. Rangers check permits at the trailhead on weekends.")
doc("h11", "Blister prevention", "Tape hot spots early, wear wool socks, and lace the heel lock. Wet skin blisters faster so change socks at lunch.")
doc("h12", "Navigation basics", "Orient the map to north with a compass and tick off features as you pass. GPS batteries die, paper does not.")

# --- machine learning -------------------------------------------------------
doc("m01", "What is overfitting", "Overfitting is when a model memorizes training data and fails on new data. The training loss keeps falling while validation loss rises.")
doc("m02", "Precision versus recall", "Precision is the fraction of predicted positives that are correct. Recall is the fraction of true positives that are found. Raising one usually lowers the other.")
doc("m03", "Learning rate too high", "If the learning rate is too high the loss oscillates or diverges. Warmup and decay schedules keep optimization stable.")
doc("m04", "Regularization overview", "Weight decay, dropout, and early stopping all limit model capacity to reduce overfitting and improve generalization.")
doc("m05", "Train test split", "Hold out a test set that is never used for tuning. Leakage between splits inflates reported accuracy.")
doc("m06", "Batch size effects", "Large batches give smoother gradients but may generalize worse. Small batches add noise that can act as regularization.")
doc("m07", "Cross validation", "K fold cross validation rotates the held out fold so every example is tested once, giving a better estimate on small datasets.")
doc("m08", "Class imbalance", "With rare positives, accuracy is misleading. Use precision, recall, and resampling or class weights instead.")
doc("m09", "Feature scaling", "Standardize features to zero mean and unit variance so gradient descent converges faster and distances are comparable.")
doc("m10", "Confusion matrix", "A confusion matrix counts true positives, false positives, true negatives, and false negatives for a classifier.")
doc("m11", "Early stopping", "Stop training when validation loss stops improving for several epochs and keep the best checkpoint.")
doc("m12", "Gradient clipping", "Clip the gradient norm to a threshold to prevent exploding gradients in recurrent networks and transformers.")

Q = [
 {"qid": "q01", "qtype": "paraphrase", "text": "how do i descale my coffee machine", "relevant": ["c03", "c05"]},
 {"qid": "q02", "qtype": "exact-id",   "text": "what does error e-4417 mean", "relevant": ["c07"]},
 {"qid": "q03", "qtype": "negation",   "text": "hiking trails that are not steep", "relevant": ["h02", "h06"]},
 {"qid": "q04", "qtype": "topical",    "text": "best water temperature for brewing coffee", "relevant": ["c02"]},
 {"qid": "q05", "qtype": "topical",    "text": "how should i store coffee beans", "relevant": ["c06"]},
 {"qid": "q06", "qtype": "topical",    "text": "what to pack for a day hike", "relevant": ["h03"]},
 {"qid": "q07", "qtype": "topical",    "text": "signs of altitude sickness", "relevant": ["h04"]},
 {"qid": "q08", "qtype": "paraphrase", "text": "model does great on training data but badly on new data", "relevant": ["m01", "m04"]},
 {"qid": "q09", "qtype": "topical",    "text": "difference between precision and recall", "relevant": ["m02"]},
 {"qid": "q10", "qtype": "paraphrase", "text": "loss keeps jumping around during training", "relevant": ["m03", "m12"]},
]

with open("docs.jsonl", "w") as f:
    for d in D: f.write(json.dumps(d) + "\n")
with open("queries.jsonl", "w") as f:
    for q in Q: f.write(json.dumps(q) + "\n")
print(f"{len(D)} docs, {len(Q)} queries")
