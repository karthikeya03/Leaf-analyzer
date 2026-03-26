import re

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace everything from <style> to </style> inclusive with the link tag
pattern = re.compile(r'<style>.*?</style>', re.DOTALL)
new_content = pattern.sub('<link rel="stylesheet" href="/static/premium.css" />', content)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Replaced style block with external CSS link.")
