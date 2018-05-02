import re
from collections import Counter


a = ["a", "a", "b", "c"]
cnt = Counter()
cnt.update(a)
print(cnt["c"])