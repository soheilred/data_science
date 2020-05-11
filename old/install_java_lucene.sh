#!/bin/bash

[link](https://www.2daygeek.com/install-java-openjdk-jre-7-8-10-11-on-ubuntu-centos-debian-fedora-archlinux/)

pacman -Ss java | grep openjdk

# extra/jdk8-openjdk 8.u232-1 [installed]
# extra/jre8-openjdk 8.u232-1 [installed]
# extra/jre8-openjdk-headless 8.u232-1 [installed]
# extra/openjdk8-doc 8.u232-1 [installed]
# extra/openjdk8-src 8.u232-1 [installed]

sudo pacman -Syu jdk8-openjdk

export JAVA_HOME="/usr/lib/jvm/java-8-openjdk"
export JRE_HOME="/usr/lib/jvm/java-8-openjdk/jre/bin/java"
PATH=$PATH:$HOME/bin:JAVA_HOME:JRE_HOME

sudo pacman -S maven --noconfirm

# JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64/bin/java"
# JRE_HOME="/usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java"
# PATH=$PATH:$HOME/bin:JAVA_HOME:JRE_HOME

sudo source ~/.bashrc


# Add these to the .pom file

    <dependency>
      <groupId>org.apache.lucene</groupId>
      <artifactId>lucene-core</artifactId>
      <version>8.4.1</version>
      <scope>system</scope>
      <systemPath>${basedir}/lib/lucene-core-8.4.1.jar</systemPath>
    </dependency>

    <dependency>
      <groupId>org.apache.lucene</groupId>
      <artifactId>lucene-queryparser</artifactId>
      <version>8.4.1</version>
      <scope>system</scope>
      <systemPath>${basedir}/lib/lucene-queryparser-8.4.1.jar</systemPath>
    </dependency>