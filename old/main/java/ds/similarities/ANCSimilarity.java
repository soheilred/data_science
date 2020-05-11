package edu.unh.cs.ir.similarities;

import org.apache.lucene.search.similarities.BasicStats;
import org.apache.lucene.search.similarities.SimilarityBase;

public class ANCSimilarity extends SimilarityBase {

    public ANCSimilarity() {

    }

    @Override
    protected float score(BasicStats stats, float freq, float docLen) {
        //float max = Collections.max(stats.getTotalTermFreq());
        float a = 0.5f + (0.5f * stats.getTotalTermFreq()); //TODO
        float n = 1.0f;
        float c = stats.getValueForNormalization();
        return a * n * c;
    }

    @Override
    public String toString() {
        return null;
    }
}