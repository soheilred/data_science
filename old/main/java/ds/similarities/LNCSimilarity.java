package edu.unh.cs.ir.similarities;

import org.apache.lucene.search.similarities.BasicStats;
import org.apache.lucene.search.similarities.SimilarityBase;

import static java.lang.Math.log10;

public class LNCSimilarity extends SimilarityBase {


    public LNCSimilarity() {

    }


    @Override
    protected float score(BasicStats stats, float freq, float docLen) {

        float l = (float) (1 + log10(stats.getTotalTermFreq()));
        float n = 1.0f;
        float c = stats.getValueForNormalization();

        return (l * n * c);
    }

    @Override
    public String toString() {
        return null;
    }
}