package edu.unh.cs.ir.pa5;

import co.nstant.in.cbor.CborException;
import edu.unh.cs.treccar.Data;
import edu.unh.cs.treccar.read_data.DeserializeData;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.similarities.SimilarityBase;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;


public class MyIndexer {

    public MyIndexer () {
    }

    private IndexWriter indexWriter;

    public void buildIndexes(FileInputStream fileInputStream, SimilarityBase similarity) throws IOException, CborException {
        for (Data.Paragraph paragraph : DeserializeData.iterableParagraphs(fileInputStream)) {
            // Index all Accommodation entries
            if (indexWriter == null) {
                Directory indexDir = FSDirectory.open(new File("index-directory5").toPath());
                IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());
                if (similarity != null) {
                    config.setSimilarity(similarity);
                }
                indexWriter = new IndexWriter(indexDir, config);
            }
            IndexWriter writer = indexWriter;

            Document doc = new Document();
            doc.add(new StringField("id", paragraph.getParaId(), Field.Store.YES));
            doc.add(new TextField("content", paragraph.getTextOnly(), Field.Store.YES));
//            System.out.println(doc.toString());

            writer.updateDocument(new Term("id", paragraph.getParaId()), doc);
        }
        System.out.print(indexWriter.numDocs());

        if (indexWriter != null) {
            indexWriter.close();
        }
    }

}