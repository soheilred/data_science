package edu.unh.cs.ir.similarities;

import org.apache.lucene.search.similarities.BasicStats;
import org.apache.lucene.search.similarities.SimilarityBase;

public class BNNSimilarity extends SimilarityBase {


    public BNNSimilarity() {

    }


    @Override
    protected float score(BasicStats stats, float freq, float docLen) {
        float b;
        if (stats.getTotalTermFreq() > 0)
            b = 1.0f;
        else
            b = 0.0f;
        float n = 1.0f;

        return (b * n * n);
    }

    @Override
    public String toString() {
        return null;
    }
}