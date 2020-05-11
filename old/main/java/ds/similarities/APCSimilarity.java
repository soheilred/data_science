package edu.unh.cs.ir.similarities;

import org.apache.lucene.search.similarities.BasicStats;
import org.apache.lucene.search.similarities.SimilarityBase;

import static java.lang.Math.log10;

public class APCSimilarity extends SimilarityBase {

    public APCSimilarity() {

    }

    @Override
    protected float score(BasicStats stats, float freq, float docLen) {
        float a = 0.5f + (0.5f * stats.getTotalTermFreq()); //TODO
        float p = Math.max(0, (float) log10( (stats.getNumberOfDocuments() - stats.getDocFreq() ) / (stats.getDocFreq())));
        float c = stats.getValueForNormalization();

        return a * p * c;
    }

    @Override
    public String toString() {
        return null;
    }
}