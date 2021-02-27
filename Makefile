komp:
	gcc -o komp komp.c num.c

ext:
	gcc -arch x86_64 -fPIC -shared -o lib.so komp.c num.c

clean:
	rm komp lib.so 
