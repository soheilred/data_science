package edu.unh.cs.ir.similarities;

import org.apache.lucene.search.similarities.BasicStats;
import org.apache.lucene.search.similarities.SimilarityBase;

import static java.lang.Math.log10;

public class LTNSimilarity extends SimilarityBase {


    public LTNSimilarity() {

    }


    @Override
    protected float score(BasicStats stats, float freq, float docLen) {
        float l = (float) (1 + log10(stats.getTotalTermFreq()));
        float t = (float) log10((stats.getNumberOfDocuments()) / (stats.getDocFreq()));
        float n = 1.0f;
        return (l * t * n);
    }

    @Override
    public String toString() {
        return null;
    }
}