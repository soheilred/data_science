package edu.unh.cs.ir.similarities;

import org.apache.lucene.search.similarities.BasicStats;
import org.apache.lucene.search.similarities.SimilarityBase;

public class LMLaplaceSimilarity extends SimilarityBase {

    private long vocabSize;
    private boolean isDebug;

    public LMLaplaceSimilarity(int vocabSize, boolean isDebug) {
        this.vocabSize = vocabSize;
        this.isDebug = isDebug;
    }


    @Override
    protected float score(BasicStats stats, float freq, float docLen) {
        if (isDebug) {
            System.out.printf("\nTotalTermFreq = " + String.valueOf(stats.getTotalTermFreq()));
            System.out.println("\nNumberOfDocuments = " + String.valueOf(stats.getNumberOfDocuments()));
            System.out.println("ValueForNormalization = " + String.valueOf(stats.getValueForNormalization()));
            System.out.println("AvgFieldLength = " + String.valueOf(stats.getAvgFieldLength()));
            System.out.println("NumberOfFieldTokens = " + String.valueOf(stats.getNumberOfFieldTokens()));
            System.out.println("Boost = " + String.valueOf(stats.getBoost()));
            System.out.println("docLen: " + String.valueOf(docLen));
            System.out.println("freq: " + String.valueOf(freq));
            System.out.println("DocFreq = " + String.valueOf(stats.getDocFreq()) + "\n");
        }
        return (freq + 1) / (docLen + vocabSize);
    }

    @Override
    public String toString() {

        return null;
    }
}