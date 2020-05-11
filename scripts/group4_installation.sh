#!/bin/bash

BOOKSDIR="data/"



if [[ $1 == "run" ]]; then

	cd scripts

    python3 evaluation.py create_feature add_feature

    python3 evaluation.py create_feature

elif [[ $1 == "setup" ]]; then
	sudo pip install nltk numpy pandas sklearn seaborn matplotlib

	python3 scripts/nltk_installation.py

	if [[ ! -d $BOOKSDIR ]]
	then
		echo "please put books in a folder called data"
		exit
	fi

	echo "cleaning non-ascii characters from the books"
	cd $BOOKSDIR
	iconv -f ISO-8859-1 -t utf-8 -c 004ssb.txt -o tmp.txt
	rm 004ssb.txt
	mv tmp.txt 004ssb.txt

	iconv -f ISO-8859-1 -t utf-8 -c 005ssb.txt -o tmp.txt
	rm 005ssb.txt
	mv tmp.txt 005ssb.txt

	cd ..

	# if [[ ! -d "files" ]]
	# then
	# 	echo "please put character-death.csv in a folder called files"
	# 	exit
	# fi

fi

# echo -n "Enter the name of a country: "
# read COUNTRY
# # if [[ $# -eq 0  ]]; then

