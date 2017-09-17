all: README README.org

# a multiple-target pattern rule means that a single invocation of the command
# builds all the targets, which is what I want here
%EADME %EADME.org: numpysane.py README.footer.org extract_README.py
	python extract_README.py numpysane

test:  test2  test3
check: check2 check3
check2: test2
check3: test3
test2 test3:
	python$(patsubst test%,%,$@) test_numpysane.py
.PHONY: check check2 check3 test test2 test3

# make distribution tarball
dist:
	python setup.py sdist
.PHONY: dist

# make and upload the distribution tarball
dist_upload:
	python setup.py sdist upload
.PHONY: dist_upload

clean:
	rm -f README.org README
.PHONY: clean

