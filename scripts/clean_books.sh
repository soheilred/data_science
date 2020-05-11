#!/bin/bash

BOOKSDIR="../data/"


if [[ ! -d $BOOKSDIR ]]
then
	echo "please put books in a folder called data"
	exit
fi

echo "cleaning non-ascii characters from the books"
cd $BOOKSDIR
# iconv -f ISO-8859-1 -t utf-8 -c 004ssb.txt -o tmp.txt
perl -pe's/[[:^ascii:]]//g' < 004ssb.txt > tmp.txt
rm 004ssb.txt
mv tmp.txt 004ssb.txt

perl -pe's/[[:^ascii:]]//g' < 005ssb.txt > tmp.txt
# iconv -f ISO-8859-1 -t utf-8 -c 005ssb.txt -o tmp.txt
rm 005ssb.txt
mv tmp.txt 005ssb.txt


