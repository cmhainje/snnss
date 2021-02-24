komp:
	gcc -o komp komp.c num.c

ext:
	gcc -fPIC -shared -o lib.so komp.c num.c

clean:
	rm komp lib.so 