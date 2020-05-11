package ds.pa5;

import co.nstant.in.cbor.CborException;
import edu.unh.cs.treccar.Data;
import edu.unh.cs.treccar.read_data.DeserializeData;
import org.apache.lucene.document.Document;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;

import java.io.*;
import java.lang.reflect.Array;
import java.util.*;

public class Assignment5 {

    public void task1() {
//        String qId;
//        String dId;
//
//        try {
//            BufferedWriter bw = new BufferedWriter(new FileWriter("outputs/pa5/T1RLFeatures"));
//
//            String[] runfileFuncs = {"outputs/pa5/rankfile1", "outputs/pa5/rankfile2", "outputs/pa5/rankfile3", "outputs/pa5/rankfile4"};
//
//            ArrayList<String> rankLibStr = new ArrayList<>();
//            int rank = 0;
//
//            float feature = 0;
//            String featureStr = "";
//            int target = 0;
//            qId = "Q";
//            for (int j = 0; j < 12; j++) {
//                dId = "D" + String.valueOf(j + 1);
//                featureStr = "";
//                for (int i = 0; i < runfileFuncs.length; i++) {
//                    rank = rankParser(runfileFuncs[i]);
//                    if (rank > 0) {
//                        feature = (1 / (float) rank);
//                    } else {
//                        feature = 0;
//                    }
//                    featureStr = featureStr.concat(" " + (i + 1) + ":" + String.format("%.2f", feature));
//                    target = targetParser("outputs/pa5/T1qrelfile", dId, qId);
//                }
//                rankLibStr.add(target + " qid:" + qId + featureStr + " # " + dId);
//            }
//
//            for (String str : rankLibStr) {
//                bw.write(str + "\n");
//            }
//            bw.close();
//        } catch (Exception e) {
//            System.out.println("Exception caught." + e.toString() + "\n");
//        }
//

    }

    public void task2() {
        String qId;
        String dId;

        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter("RankLibFileTask2"));

            String[] runfileFuncs = {"outputs/pa5/T2rankfiles/lnc_ltn", "outputs/pa5/T2rankfiles/bnn_bnn", "outputs/pa5/T2rankfiles/LM_U", "outputs/pa5/T2rankfiles/U_JM", "outputs/pa5/T2rankfiles/U_DS"};

            ArrayList<String> rankLibStr = new ArrayList<>();

            // read the queries' file
            File fOutlines = new File("./test200/train.test200.cbor.outlines");
            final FileInputStream fISOutlines = new FileInputStream(fOutlines);

            // read the paragraphs' file
            File fParags = new File("./test200/train.test200.cbor.paragraphs");
            final FileInputStream fISParags = new FileInputStream(fParags);

            ////////////////
            int n = 0;
            Map<Integer, String > docID = new HashMap<>();
            for (Data.Paragraph paragraph : DeserializeData.iterableParagraphs(fISParags)) {
                docID.put(n, paragraph.getParaId());
                n++;
            }
            int docSize = n;
            /// make ranking hashmaps ///
            List<List<Map<String, Map<String, Integer>>>> rankLists = new ArrayList<>();
            for (int i = 0; i < runfileFuncs.length; i++) {
                List<Map<String, Map<String, Integer>>> myrank;
                myrank = rankParser(runfileFuncs[i]);
                rankLists.add(myrank);
            }
            Map<String , String> targetMap = targetParser("./test200/train.test200.cbor.article.qrels");

            ////////////////
            int ranknum = 0;
            for (List<Map<String, Map<String, Integer>>> diffRanks : rankLists){
                for (Map<String, Map<String, Integer>> line : rankLists.get(ranknum)){

                }
                ranknum++;
            }
            //////////

            float feature = 0;
            int target = 0;
//            int rank = 0;
//            for (Data.Page page : DeserializeData.iterableAnnotations(fISOutlines)) {
//                qId = page.getPageId();
////                for (Data.Paragraph paragraph : DeserializeData.iterableParagraphs(fISParags)) {
//                for (int m = 0; m < docSize; m++){
//                    dId = docID.get(m);
//                    String featureStr = "";
//                    for (int i = 0; i < runfileFuncs.length; i++) {
//                        Map<String, Integer> myrank = rankLists.get(i);
//                        RankClass a = myrank.get(dId);
//                        rank = a.getRank();
//                        if (rank > 0) {
//                            feature = (1 / (float) rank);
//                        } else {
//                            feature = 0;
//                        }
//                        featureStr = featureStr.concat(" " + (i + 1) + ":" + String.format("%.2f", feature));
//                        Map<String, String > mytarget = targetMap;
//                        mytarget.containsKey(qId);
//                    }
//                    rankLibStr.add(target + " qid:" + qId + featureStr + " # " + dId);
//                }
//            }
//
//            for (String str : rankLibStr) {
//                bw.write(str + "\n");
//            }
            bw.close();
        } catch (Exception e) {
            System.out.println("Exception caught." + e.toString() + "\n");
        }

    }

    public List<Map<String, Map<String, Integer>>> rankParser(String rf) {
        List<Map<String, Map<String, Integer>>> rankMap = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(rf));
            String line;
            String[] linesArray;
            while ((line = br.readLine()) != null) {
                linesArray = line.split(" ");
                Map<String, Integer> docAndRank= new HashMap<>();
                docAndRank.put(linesArray[2], Integer.parseInt(linesArray[3]));
                Map<String , Map<String , Integer>> qDocAndRank = new HashMap<>();
                qDocAndRank.put(linesArray[0], docAndRank);
                rankMap.add(qDocAndRank);
                }
            br.close();
        } catch (Exception e) {
            System.out.println("Run File Parser Exception Caught." + e.toString() + "\n");
        }
        return rankMap;
    }

    public HashMap<String, String> targetParser(String qrelFile) {
        HashMap<String, String> rankMap = new HashMap<>();

        try {
            BufferedReader br = new BufferedReader(new FileReader(qrelFile));
            String line;
            String[] linesArray;
            while ((line = br.readLine()) != null) {
                linesArray = line.split(" ");
                rankMap.put(linesArray[0], linesArray[2]);
                }


            br.close();
        } catch (Exception e) {
            System.out.println("Target File Parser Exception Caught." + e.toString() + "\n");
        }
        return rankMap;
    }

//    public int targetParser(String qrelFile, String doc, String query) {
//        int target = 0;
//        try {
//            BufferedReader br = new BufferedReader(new FileReader(qrelFile));
//            String line;
//            String[] linesArray;
//            while ((line = br.readLine()) != null) {
//                linesArray = line.split(" ");
//                if (linesArray[0].equals(query) && linesArray[2].equals(doc)) {
//                    target = 1;
//                    break;
//                }
//            }
//            br.close();
//        } catch (Exception e) {
//            System.out.println("Target Parser Exception Caught." + e.toString() + "\n");
//        }
//        return target;
//    }

    public static void main(String[] args) throws FileNotFoundException, CborException {
        int taskNumber = 2; //TODO: change this to run for the desired task
        Assignment5 a5 = new Assignment5();

        if (taskNumber == 1) {
            a5.task1();
        } else {
            a5.task2();
        }
    }

}
