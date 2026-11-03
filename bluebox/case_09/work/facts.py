from common import *
m = load(); tok=tools.tok
ps = ["The Eiffel Tower is in", "The Eiffel Tower is located in the city of", "The Colosseum is located in the city of","The Statue of Liberty is located in the city of","Big Ben is located in the city of",
"The Louvre is located in the city of","Paris is the capital of","The capital of France is the city of","The Eiffel Tower, located in",
"Water freezes at a temperature of", "The Great Wall is located in", "Michael Jordan plays the sport of","Lionel Messi plays the sport of",
"The Space Needle is located in the city of", "The mother tongue of Danielle Darrieux is", "Steve Jobs was the CEO of","The iPhone is made by",
"The Taj Mahal is located in", "Mount Everest is located in", "The Kremlin is located in the city of", "The Golden Gate Bridge is in the city of"]
for p in ps:
    lg = m(**tok(p,return_tensors="pt")).logits[0,-1].softmax(-1); t=lg.topk(5)
    print(repr(p), [(tok.decode([i]), round(v,3)) for v,i in zip(t.values.tolist(), t.indices.tolist())])
