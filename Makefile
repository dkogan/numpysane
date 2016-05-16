all: README README.org

# a multiple-target pattern rule means that a single invocation of the command
# builds all the targets, which is what I want here
%EADME %EADME.org: numpysane.py README.header.org README.footer.org extract_README.py
	python extract_README.py

clean:
	rm -f README.org README
.PHONY: clean

