all: README README.org

# a multiple-target pattern rule means that a single invocation of the command
# builds all the targets, which is what I want here
%EADME %EADME.org: numpysane.py README.header.org README.footer.org extract_README.py
	python extract_README.py

check: check2 check3
test:  test2  test3
test2 test3 check2 check3:
	python$(patsubst test%,%,$@) test_numpysane.py
.PHONY: check check2 check3 test test2 test3

clean:
	rm -f README.org README
.PHONY: clean

