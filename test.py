from textblob.classifiers import NaiveBayesClassifier

train = [
    ('I love this sandwich.', 'pos'),
    ('this is an amazing place!', 'pos'),
    ('I feel very good about these beers.', 'pos'),
    ('this is my best work.', 'pos'),
    ("what an awesome view", 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
    ('he is my sworn enemy!', 'neg'),
    ('my boss is horrible.', 'neg'),
    ('Nak, Help yourself to be happy, don’t feed your mind with negative thoughts.', 'pos'),
    (
        'Holding yourself from not sliding into negative thoughts or negative emotions is a full-time job, and if you take a break, you break.',
        'pos'),
    ("This is an amazing library!", "pos"),
    (
        "Ah, that sweet Miami moment when a drawbridge has just closed, cyclists/runners dart under the crossing gates before the cars can get thru, & for 15 sweet seconds it's just us on the road, as if we actually had infrastructure that proactively enabled safe non-car-transportation",
        "neg"),
    (
    "only FIVE women have ever been nominated for the Best Director spot in @TheAcademy 92 year history and improvement is long due.",
    "neg")
]
test = [
    ('the beer was good.', 'pos'),
    ('I do not enjoy my job', 'neg'),
    ("I ain't feeling dandy today.", 'neg'),
    ("I feel amazing!", 'pos'),
    ('Gary is a friend of mine.', 'pos'),
    ("I can't believe I'm doing this.", 'neg')
]
data = [
    "the freezing great lakes are the most beautiful thing in nature, don't @ me",
    "I'm headed for a new adventure! This week, I’m resigning from Netflix to return to being a software engineer. I’ll announce where I’m moving to next month.",
    "Some languages are great for programming but make me cringe as a language designer (e.g., Go). Some are beautiful to a designer but I wouldn't want to do real work with them (e.g., Haskell, don't @ me). I really like Rust because to me it is one of the few that is good at both",
    "Negative. Vindman didn’t use his chain of command to notify his supervisor. Instead he went to the whistleblower outside the NSC & then lied under oath about knowing the whistleblower’s ID. He failed himself, his Chain of Command & his Commander in Chief.",
    "Continue  surrounding your bed with teddy bear ..till one of them will touch you one night and say I want to pee",
    "Trying to clean house - damn you art of tidying! Watching @MarriageStory - what a monotonous droning boring shit-fest. An hour in I'm done. Do they get divorced? I don't care! Even as background noise it's annoying. #Oscars are such a high-school popularity contest. 🙄",
    "Finally got round to watching #HairLove. It's an important story by @MatthewACherry , so beautifully told. And it won the best animated short film at the #Oscars too!",
    "Due to self-complacency and lack of ambitions #Bollywood fails to inspire. #Parasite",
    "The logical question is: What did #Parasite do that Indian movies fail to do to win at big international events?",
    "Hold on now. You mean to tell me the #Oscars set off a social media rage? I am so shocked! How? What? Why?",
    "Well done #Oscars for picking yet another movie that most of us movie regular people will go and see (sarcasm me thinks!). Not trying to knock it as I haven't seen it but judging from most feedback, either of 1917 or Joker was robbed. It's Moonlight & Green Book all over again.!",
    "Is #Parasite awarded everything in #Oscars fearing the #coronavirus outbreak?"
]

cl = NaiveBayesClassifier(train)
acc = cl.accuracy(test)
print(f"ACCURACY: {acc}")
for phrase in data:
    result = cl.classify(phrase)
    print(f"RESULT: {phrase}:{result}\n")
    # prob_dist = cl.prob_classify("This one's a doozy.")
