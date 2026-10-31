import sys,re
t=open(sys.argv[1],encoding='utf-8',errors='replace').read()
for a,b in {'ﬁ':'fi','ﬂ':'fl','ﬀ':'ff','ﬃ':'ffi','ﬄ':'ffl','−':'-','–':'-','—':'-','’':"'",'“':'"','”':'"'}.items(): t=t.replace(a,b)
t=re.sub(r'\f','',t); t=re.sub(r'(\w)-\s*\n\s*(\w)',r'\1\2',t)  # dehyphenate
t=re.sub(r'-','',t)
w=re.findall(r'[A-Za-z0-9.,%/]+',t)
# drop page numbers lines (pure digits) to reduce noise
print('\n'.join(x for x in w if not re.fullmatch(r'\d{1,2}',x)))
