package monclubelec.javacvPro;

//======= import =========
//--- processing --- 
import processing.core.*;

//--- javacv / javacpp ---- 
import com.googlecode.javacpp.*;
import com.googlecode.javacv.*;
import com.googlecode.javacv.cpp.opencv_calib3d;
import com.googlecode.javacv.cpp.opencv_core;
import com.googlecode.javacv.cpp.opencv_features2d;
import com.googlecode.javacv.cpp.opencv_highgui;
import com.googlecode.javacv.cpp.opencv_imgproc;
import com.googlecode.javacv.cpp.opencv_objdetect;
import com.googlecode.javacv.cpp.opencv_video;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

import jp.nyatla.nyar4psg.*; // ARToolkit for Processing 


//---- java ---- 
import java.nio.*; // pour classe ByteBuffer
import java.awt.*; // pour classes Point , Rectangle..
import java.awt.image.BufferedImage;
import java.util.ArrayList; // pour ArrayList
import java.util.List;
import java.math.*; // fonctions math

import java.io.File;



// JavacvPro library for Processing 
// Written by X. HINAULT - www.mon-club-elec.fr
// all rights reserved - 09/2011
//This code is under license GPL v3

// Credits : 
// Tested with Opencv 2.3.1 version http://opencv.willowgarage.com/wiki/
// Based on javacv by Samuel Audet - http://code.google.com/p/javacv/
// Inspired by OpenCV library for Processing - http://ubaa.net/shared/processing/opencv/
// About Processing, see : http://processing.org/


/* Pour la plupart des fonctions de traitement d'image de la classe OpenCV 
 * il existe une fonction principale dans la forme la plus complète et contenant le code effectif principal
 * les autres formes disponibles de la fonction sont basées sur la fonction principale
 */



public class OpenCV { // crée la classe

	
	/////////////// Les variables ///////////////////////
	
	//------------------ variables d'instances

	//--- constantes de classe

	//--- champs pour identifiant les Buffer (cf selectBuffer() --- 
	public static final String BUFFER="BUFFER";
	public static final String GRAY="GRAY";
	public static final String RED="RED";
	public static final String GREEN="GREEN";
	public static final String BLUE="BLUE";
	public static final String MEMORY="BLUE";
	public static final String MEMORY2="MEMORY2";

	//--- champs pour fonction Flip() --- 
	public static final String VERTICAL="VERTICAL";
	public static final String HORIZONTAL="HORIZONTAL";
	public static final String BOTH="BOTH";
	
	//---- champs pour fonction SURF ----
	public static final boolean OBJECT=false;
	public static final boolean SCENE=true;
	
	//--- variables de classe
	//--- variables générales 
	static PApplet p; // représente le PApplet Processing
	private static String OS = System.getProperty("os.name").toLowerCase();

	//-- objet utiles pour mémorisation des contours déclarés en static -- 
	// à revoir ?
	static ArrayList<Blob> myBlobsList = new ArrayList<Blob>(); // ArrayList de Blob
	static opencv_core.CvMemStorage storage= opencv_core.cvCreateMemStorage(0); // initialise objet conteneur CvMemStorage - déclaré en static 
	static opencv_core.CvSeq contours = new opencv_core.CvSeq(null); // initialise objet CvSeq vide - initialiser en tant qu'objet "null" - cvFindContours renverra un cvSeq rempli
	static opencv_core.CvSeq[] contourCourant= new opencv_core.CvSeq[2500]; // initialise un tableau static d'objet CvSeq d'objet CvSeq
	
	//--- déclare objets utiles publics pour la détection de forme 
	static opencv_objdetect.CvHaarClassifierCascade cascade; 
	static ArrayList<Rectangle> myRectList = new ArrayList<Rectangle>(); // alternative à un tableau fixe = créer un ArrayList de Rectangle (une sorte de tableau à taille variable)

	
	//----------- déclare objets détecteurs utiles --------- 
	static opencv_features2d.SurfFeatureDetector detectorSURF=null;   

	//--- déclare objet utiles pour la détection de points clés
	
	static ArrayList<Keypoint> myKeypointsListObject = new ArrayList<Keypoint>(); // ArrayList de Keypoint
	static ArrayList<Keypoint> myKeypointsListScene = new ArrayList<Keypoint>(); // ArrayList de Keypoint
	//static ArrayList<Keypoint> myKeypointsList = new ArrayList<Keypoint>(); // ArrayList de Keypoint

	static opencv_features2d.KeyPoint keypointsObject = null; // vecteur de Keypoints natif openCV
	static opencv_features2d.KeyPoint keypointsScene = null; // vecteur de Keypoints natif openCV

	//--- déclare objets utiles pour la détection de cercles, lignes --- 
	static ArrayList<Circle> myCirclesList = new ArrayList<Circle>(); // ArrayList d'objet Line
	static ArrayList<Line> myLinesList = new ArrayList<Line>(); // ArrayList de d'objet Line
	
	// non... 
	//static opencv_features2d.KeyPoint keypointsObjectMatch = null; // vecteur de Keypoints natif openCV
	//static opencv_features2d.KeyPoint keypointsSceneMatch = null; // vecteur de Keypoints natif openCV

	// non.. 
	//static opencv_features2d.KeyPoint keypointsObjectGoodMatch = null; // vecteur de Keypoints natif openCV
	//static opencv_features2d.KeyPoint keypointsSceneGoodMatch = null; // vecteur de Keypoints natif openCV

	static opencv_features2d.DMatch matches=null; // vecteur de DMatch natif openCV  
	static int comptGoodMatch=0; // variable de comptage des concordances utiles 
	static float distanceMaxGoodMatch=0; // variable mémorisant le seuil distance max utilisé pour les goodMatch - voir matchSURF()
	
	static opencv_core.CvMat cvMatGoodMatchObject=null; // CvMat pour stockage des index des points GoodMatch objet
	static opencv_core.CvMat cvMatGoodMatchScene=null; // CvMat pour stockage des index des points GoodMatch Scene
	
	
	
	//----- objets utiles pour la soustraction de fond 
	public opencv_video.BackgroundSubtractorMOG bgsMOG=null; // objet MOG - cf bgsMOGInit()
	public opencv_video.BackgroundSubtractorMOG2 bgsMOG2=null; // objet MOG - cf bgsMOGInit()
	
	/**
	 * Le nombre maximum de sommets pour la détection de Blobs (vertices)
	 */
	public static final int MAX_VERTICES = 1024; 

	private int brightness	= 0; // variable de luminosité : -128 / + 128
	private int contrast	= 0; // variable de contraste : -128 / + 128

	
	//--- déclaration des iplImage qui seront initialisé via allocate
	public opencv_core.IplImage Buffer; // crée l'image Buffer

	public opencv_core.IplImage BufferR; // crée l'image Buffer R - canal rouge
	public opencv_core.IplImage BufferG; // crée l'image Buffer G - canal vert
	public opencv_core.IplImage BufferB; // crée l'image Buffer B - canal bleu

	//public opencv_core.IplImage BufferA; // crée l'image Buffer A - canal transparence

	public opencv_core.IplImage BufferGray; // crée l'image Buffer niveaux de gris 

	//--- crée 2 buffers de mémorisation d'image 
	public opencv_core.IplImage Memory; // crée l'image Buffer
	public opencv_core.IplImage Memory2; // crée l'image Buffer

	
	//--- buffers de travail intermédiaires nécessaires pour certaines opérations sur images --- 
	static opencv_core.IplImage Trans16S3C; // crée l'image de travail - 16 bits signés - 3 canaux
	static opencv_core.IplImage Trans16S3C1; // crée l'image de travail - 16 bits Signés - 3 canaux - n°2
	static opencv_core.IplImage Trans16S3C2; // crée l'image de travail - 16 bits Signés - 3 canaux - n°3

	static opencv_core.IplImage Trans16S1C; // crée l'image de travail - 16 bits Signés - 1 canal - n°1
	static opencv_core.IplImage Trans16S1C1; // crée l'image de travail - 16 bits Signés - 1 canal - n°2
	static opencv_core.IplImage Trans16S1C2; // crée l'image de travail - 16 bits Signés - 1 canal - n°3

	static opencv_core.IplImage Trans8U3C; // crée l'image de travail - 8 bits non signés - 3 canaux
	static opencv_core.IplImage Trans8U3C1; // crée l'image de travail - 8 bits non signés - 3 canaux - n°2
	static opencv_core.IplImage Trans8U3C2; // crée l'image de travail - 8 bits non signés - 3 canaux - n°3
	
	static opencv_core.IplImage Trans8U1C; // crée l'image de travail - 8 bits non signés - 1 canal 
	static opencv_core.IplImage Trans8U1C1; // crée l'image de travail - 8 bits non signés - 1 canal - n°2 
	static opencv_core.IplImage Trans8U1C2; // crée l'image de travail - 8 bits non signés - 1 canal - n°3

	static opencv_core.IplImage Trans32F3C; // crée l'image de travail - 32 bits signés - 3 canaux 

	//--- buffers de travail intermédiaires utiles pour Memory ---
	static opencv_core.IplImage Trans8U1CMemory; // crée l'image de travail - 8 bits non signés - 1 canal 
	static opencv_core.IplImage Trans8U1C1Memory; // crée l'image de travail - 8 bits non signés - 1 canal - n°2 
	static opencv_core.IplImage Trans8U1C2Memory; // crée l'image de travail - 8 bits non signés - 1 canal - n°3


	////////////////// les constructeurs /////////////
	
	public OpenCV() {
		
	}
	
	//---- le constructeur par défaut 
	public OpenCV(PApplet theParent) {
		p = theParent;
		welcome();
	}
	
	/////////////////// les méthodes de la classe /////////////////
	private void welcome() {
		System.out.println("javacvPro (Processing library) - version 0.4beta - by X.HINAULT - Janvier 2012 - www.mon-club-elec.fr - (c) all rights reserved - GPLv3 ");
		
	}
	
	//=============== Fonctions de gestion des transferts PImage <--> IplImage ============
	
	
	//======== toPImage : fonction pour récupérer un IplImage dans un PImage - forme complète =======

	public PImage toPImage(opencv_core.IplImage IplImageIn, boolean debug) { // la fonction reçoit l'objet IplImage source
	
	/* --- v1 --- lent - 100ms... !
	  PImage imgOut; // crée objet PImage

	  //---- créer un PImage de même taille que IplImage --- 
	  imgOut = p.createImage(IplImageIn.width(),IplImageIn.height(), PConstants.RGB); // RGB = 3 canaux 8 bits

	  imgOut.loadPixels();
	  
	  //---- créer un objet opencv CvScalar (conteneur de 1 à 4 valeur) pour contenir valeur r,g,b d'un pixel
	  opencv_core.CvScalar toPixel= new opencv_core.CvScalar(3); // scalaire contenant le groupe de valeur r,g,b d'un pixel x,y de l'image

	  //---- récupérer les pixels de l'IplImage dans le PImage --- 

	  for (int x=0; x <imgOut.width; x++) { //-- défile les pixels de l'image
	    for (int y=0; y <imgOut.height; y++) {
	    
	    toPixel=opencv_core.cvGet2D(IplImageIn,y,x); // récupère la valeur du pixel - attention y,x et pas x,y !
	    //--- passer par cvGetCol pour accélérer ? 
	    
	    int r=(int)toPixel.red(); // récupère la valeur du canal du pixel (double) et convertit en int
	    int g=(int)toPixel.green(); // récupère la valeur du canal du pixel (double) et convertit en int
	    int b=(int)toPixel.blue(); // récupère la valeur du canal du pixel (double) et convertit en int
	    
	    imgOut.pixels[ x + (y *  imgOut.width)] = p.color(r, g, b); // modifie le pixel du PImage en fonction
	    
	    } // fin y 
	  } // fin x 

	  imgOut.updatePixels();
	  
	  */
		
	// V2 -- 09/2011 -- passe par un ByteBuffer --- plus rapide -- 4ms ! -- + gestion du nombre canaux... 
		
		  PImage imgOut; // crée objet PImage
		  
		  //---- créer un PImage de même taille que IplImage --- 
		  imgOut = p.createImage(IplImageIn.width(),IplImageIn.height(), PConstants.RGB); // RGB = 3 canaux 8 bits

		  if (debug) PApplet.println (" width =" + IplImageIn.width() + " | height="+IplImageIn.height()+ " | Taille ="  + (IplImageIn.width()*IplImageIn.height()));
		  
			//------------------------------------- si image RGB - 3 canaux ----------------------------------------------------

		if (IplImageIn.nChannels()==3) { // si l'image en réception a 3 canaux RGB
			
		  //imgOut.loadPixels();
		  
		  //---- extraction des données globales ---
		  ByteBuffer byteBuffer = IplImageIn.getByteBuffer(); // récupère les bytes de l'image dans un buffer
		  //--- la structure de l'IplImage est B0G0R0, B1G1R1, ... 
		  // --- attention : les valeurs récupérées dans le buffer sont signées... - Java ne supporte pas les valeurs non signées - voir ci-dessous
		  

		  
		  if (debug)  PApplet.println ("Capacité du buffer ="+byteBuffer.capacity()); 
		  //PApplet.println ("Limite du buffer ="+byteBuffer.limit()); 
		  //PApplet.println ("Position du buffer ="+byteBuffer.position()); 
		
		  
		  imgOut.loadPixels(); 
		  
		  //for (int i=0; i<(byteBuffer.capacity()); i=i+3){ // défile les valeurs de 3 en 3 
		  for (int i=0; i<(IplImageIn.width()*IplImageIn.height()*3); i=i+3){ // défile les valeurs de 3 en 3 
		  //for (int i=0; i<(IplImageIn.width()*IplImageIn.height()); i=i+1){ // défile les valeurs de 1 en 1 - mieux que byteBuffer.capacity()
			  
		    //imgDest.pixels[i/3]=color(byteBuffer.get(i), byteBuffer.get(i+1), byteBuffer.get(i+2) ); //=byteBuffer.get(i); 
		    // --- ne marche pas car Java ne supporte pas les non signés 
		    
			  
		     imgOut.pixels[PApplet.floor(i/3)]=p.color(byteBuffer.get(i+2)&0xFF, byteBuffer.get(i+1)&0xFF, byteBuffer.get(i)&0xFF); //=byteBuffer.get(i);  &0xFF pour rendre byte unsigned

			  //PApplet.print(" | i="+i); 
			  //imgOut.pixels[i]=p.color(byteBuffer.get((i*3)+2)&0xFF, byteBuffer.get((i*3)+1)&0xFF, byteBuffer.get(i*3)&0xFF); //=byteBuffer.get(i);  &0xFF pour rendre byte unsigned
		     // attention : inversion du bleu et du rouge entre le IplImage et le PImage... 
		     
		   	  } // fin for i 
		  
		  imgOut.updatePixels(); 
		  
		} // fin si image a 3 canaux RGB

		//------------------------------------------si image Niveaux de gris - 1 canal --------------------------------------
		
		else if (IplImageIn.nChannels()==1) { // si l'image en réception a 1 seul canal (Gray scale)
			
			  imgOut.loadPixels();
			  
			  //---- extraction des données globales ---
			  ByteBuffer byteBuffer = IplImageIn.getByteBuffer(); // récupère les bytes de l'image dans un buffer
			  //--- la structure de l'IplImage est B0G0R0, B1G1R1, ... 
			  // --- attention : les valeurs récupérées dans le buffer sont signées... - Java ne supporte pas les valeurs non signées - voir ci-dessous
			  
			  /*
			  println ("Capacité du buffer ="+byteBuffer.capacity()); 
			  println ("Limite du buffer ="+byteBuffer.limit()); 
			  println ("Position du buffer ="+byteBuffer.position()); 
				*/
			  
			  imgOut.loadPixels(); 
			  
			  for (int i=0; i<(byteBuffer.capacity()); i++){ // défile les valeurs à la suite
			    
			     imgOut.pixels[i]=p.color(byteBuffer.get(i)&0xFF, byteBuffer.get(i)&0xFF, byteBuffer.get(i)&0xFF); //=byteBuffer.get(i);  &0xFF pour rendre byte unsigned
			     // attention :  niveaux de gris donc les 3 canaux sont égaux dans le PImage 
			     
			 } // fin for i 
			  
			  imgOut.updatePixels(); 
			  
			} // fin si image a 1 canal GrayScale

	  return (imgOut); // renvoie l'objet PImage
	  

		
	} // fin toPImage

	//-------- forme sans debug -- 
	public PImage toPImage(opencv_core.IplImage IplImageIn) { // la fonction reçoit l'objet IplImage source
		
		return (	toPImage(IplImageIn, false) ); // appelle fonction complete avec debug inactif
		
	} // fin toPImage
	
	//======= fonction fromPImage : transférer un PImage dans un IplImage ===

	public opencv_core.IplImage fromPImage (PImage imgIn, opencv_core.IplImage iplImgDestIn) { // la fonction reçoit le PImage et renvoie un IplImage dans le iplImgDest
		
		


		/*
		// --- durée = 34 ms.. à améliorer.. via cvArr put et/ou buffer ?
		// ou copyfrom (bufferedImage) et setRGB de buffered image ? 
		
	  //----- créer un IplImage de même taille que le PImage --- 
	  opencv_core.CvSize fromSize=opencv_core.cvSize(imgIn.width,imgIn.height); // crée objet CvSize  - objet regroupant 2 valeurs w,h
	  opencv_core.IplImage iplImgOut= opencv_core.cvCreateImage(fromSize, opencv_core.IPL_DEPTH_8U, 3); // crée une image IplImage , 3 canaux

	  //---- créer un objet opencv CvScalar (conteneur de 1 à 4 valeur) pour contenir valeur r,g,b d'un pixel
	  opencv_core.CvScalar fromPixel = new opencv_core.CvScalar(3); // scalaire contenant le groupe de valeur r,g,b d'un pixel de l'image 
	  
	  float r=0; 
	  float g=0; 
	  float b=0; 
	  
	  //------ récupérer les pixels du PImage source dans l'IplImage destination
	  
	  for (int x=0; x <iplImgOut.width(); x++) { //-- défile les pixels de l'objet IplImage
	    for (int y=0; y <iplImgOut.height(); y++) {
	    
	    r=p.red(imgIn.pixels[x + (y *  imgIn.width)]); // récupère le canal rouge du Pixel du PImage
	    g=p.green(imgIn.pixels[x + (y *  imgIn.width)]); // récupère le canal rouge du Pixel du PImage
	    b=p.blue(imgIn.pixels[x + (y *  imgIn.width)]); // récupère le canal rouge du Pixel du PImage
	    
	    fromPixel.red(r); // fixe la valeur du canal rouge de l'IplImage
	    //fromPixel.red(0); // fixe la valeur du canal rouge de l'IplImage - debug

	    fromPixel.green(g); // fixe la valeur du canal vert de l'IplImage
	    //fromPixel.green(0); // fixe la valeur du canal vert de l'IplImage- debug

	    fromPixel.blue(b); // fixe la valeur du canal bleu de l'IplImage
	    //fromPixel.blue(0); // fixe la valeur du canal bleu de l'IplImage -debug
	       
	     //cvSet2D(opencv_core.CvArr arr, int idx0, int idx1, opencv_core.CvScalar value) 
	     opencv_core.cvSet2D(iplImgOut, y, x, fromPixel); // fixe la valeur du pixel de l'IplImage à partir du scalaire r,g,b - attention y,x et non x,y
	    
	    } // fin y 
	  } // fin x 
	  
	  */
	
	//--- fromImage v2 : 28ms - = idem en 1 peu mieux... (et meme 17ms en application compilée)
	// suite essai : cf BufferedImage est trop long (50ms...) 
		
		  //----- créer un IplImage de même taille que le PImage --- 
		  //opencv_core.CvSize fromSize=opencv_core.cvSize(imgIn.width,imgIn.height); // crée objet CvSize  - objet regroupant 2 valeurs w,h
		  //opencv_core.IplImage iplImgOut= opencv_core.cvCreateImage(fromSize, opencv_core.IPL_DEPTH_8U, 3); // crée une image IplImage , 3 canaux
		//-- utiliser buffer pour ne pas allouer de la mémoire en permanence ? 
		  
		  //---- créer un objet opencv CvScalar (conteneur de 1 à 4 valeur) pour contenir valeur r,g,b d'un pixel
		  //opencv_core.CvScalar fromPixel = new opencv_core.CvScalar(3); // scalaire contenant le groupe de valeur r,g,b d'un pixel de l'image 

		  //int indice;
		  //int indiceTrans; 
		  
		  //------ récupérer les pixels du PImage source dans l'IplImage destination
		  
		  imgIn.loadPixels();

	/*
		  // ----- cette séquence utilise width * height multiplications  soit 76800 pour 320 x 240 ! 
		   
		  for (int x=0; x<imgIn.width; x++) { // défile les pixels de l'image
		      for (int y=0; y<imgIn.height; y++) { // défile les pixels de l'image

		      indice=x + (y*imgIn.width); // calcul indice du pixel courant
		      
		      //---- récupérer les valeurs du scalaire rgb du pixel 
		      fromPixel.blue(p.blue(imgIn.pixels[indice])); 
		      fromPixel.red(p.red(imgIn.pixels[indice])); 
		      fromPixel.green(p.green(imgIn.pixels[indice])); 
		      
		      //--- modifier le pixel de l'IplImage en conséquence
		      //opencv_core.cvSet2D(iplImgOut,y,x,fromPixel); // attention inversion x,y
		      opencv_core.cvSet2D(iplImgDest,y,x,fromPixel); // attention inversion x,y
		  
		    } // fin y
		  } // fin x
		 
*/	

		  /*
		  //--- en inversant les boucles on réduit le nombre de multiplications à height !
		  
		 for (int y=0; y<imgIn.height; y++) { // défile les pixels de l'image

		    	  indiceTrans=(y*imgIn.width); // calcul intermédiaire 
		    	 
			 for (int x=0; x<imgIn.width; x++) { // défile les pixels de l'image

		      indice=x + indiceTrans; // calcul indice du pixel courant
		      
		      //---- récupérer les valeurs du scalaire rgb du pixel 
		      fromPixel.blue(p.blue(imgIn.pixels[indice])); 
		      fromPixel.red(p.red(imgIn.pixels[indice])); 
		      fromPixel.green(p.green(imgIn.pixels[indice])); 
		      
		      //--- modifier le pixel de l'IplImage en conséquence
		      //opencv_core.cvSet2D(iplImgOut,y,x,fromPixel); // attention inversion x,y
		      opencv_core.cvSet2D(iplImgDest,y,x,fromPixel); // attention inversion x,y

				  } // fin x

		    } // fin y
		      
  
		  //------- fin fromPImage v2
		  */
		  
		  // v3 : via a Buffered Image --- Yes !! - 08 Mars 2012 - only 2 ms !!
		  // simplest is the best !
		  
		  BufferedImage imgBuf= (BufferedImage) imgIn.getImage();
		  
		  opencv_core.IplImage iplImgOut=opencv_core.IplImage.createFrom(imgBuf); // crée l'image à partir du buffered Image 
		  
		  //--- extraction des 3 canaux RGB --- 
		  // le buffered image et donc iplImgOutpeut être 3 ou 4 canaux... 
		  
		  //PApplet.println("nombre canaux=" + iplImgOut.nChannels()); // debug
		  
		  //------ si image idem image Buffer ------
		  if (Buffer!=null) { // si Buffer existe
			  
		  if ((iplImgOut.nChannels()==4) // si 4 canaux = format ARGB => on va extraire RGB et éliminer le A
				  && (iplImgOut.width()==Buffer.width()) // et si largeur image == largeur buffer
				  && (iplImgOut.height()==Buffer.height()) // et si hauteur image == hauteur buffer				  
				  ){
			  
			  opencv_core.cvSplit(iplImgOut, Trans8U1C, Trans8U1C1, Trans8U1C2, null); // extrait les 3 canaux R G B : attention inversion RGB  
			  //opencv_core.cvSplit(iplImgOut, BufferB, BufferG, iplImgOut, iplImgOut); // extrait les 3 canaux R G B : attention inversion RGB

				 opencv_core.cvMerge( Trans8U1C,Trans8U1C1,Trans8U1C2, null, iplImgDestIn); // reconstruction IplImage destination à partir des canaux départ.. - attention inversion RGB
		  
				// cf mixChannels.. 
		  
		  } // fin si image idem buffer

		  } // fin si Buffer !=null
		  
		  //------ si image idem image Memory - on utilise les buffers intermédiaires Memory -----
		  if ((iplImgOut.nChannels()==4) // si 4 canaux = format ARGB => on va extraire RGB et éliminer le A
				  && (iplImgOut.width()==Memory.width()) // et si largeur image == largeur buffer
				  && (iplImgOut.height()==Memory.height()) // et si hauteur image == hauteur buffer				  
				  ){
			  
			  opencv_core.cvSplit(iplImgOut,Trans8U1CMemory, Trans8U1C1Memory, Trans8U1C2Memory, null); // extrait les 3 canaux R G B : attention inversion RGB  
			  //opencv_core.cvSplit(iplImgOut, BufferB, BufferG, iplImgOut, iplImgOut); // extrait les 3 canaux R G B : attention inversion RGB

				 opencv_core.cvMerge( Trans8U1CMemory, Trans8U1C1Memory, Trans8U1C2Memory, null, iplImgDestIn); // reconstruction IplImage destination à partir des canaux départ.. - attention inversion RGB
		  
				// cf mixChannels.. 
		  
		  } // fin si image idem Memory

	
			 iplImgOut.release(); // détruite le iplImgOut intermédiaire
			 
	  return(iplImgDestIn); 
	  
	} // fin de la fonction fromPImage
	
	//--- fonction fromPImage par défaut = utilise Buffer
	public opencv_core.IplImage fromPImage (PImage imgIn) { // la fonction reçoit le PImage et renvoie un IplImage dans le iplImgDest
		
		return (fromPImage (imgIn, Buffer));
		
	}
	
	//============================ Fonctions de gestion du/des buffers images ==================================

	//==== fonction allocate(w,h) : crée les images Iplimage utilisées comme buffers et Memory ===
	public void allocate (int widthIn, int heightIn, boolean buffers, boolean memory) {
		
		//--- la fonction reçoit 
		// la taille d'initialisation, 
		// un drapeau pour initialisation buffers, 
		// et un drapeau pour initialisation Memory
		
		// ainsi, on peut initialiser Memory et Buffers avec une taille différente... Utile pour SURF par exemple.. 
		
		//-- test si images existent --- 
		
		//if (Buffer!=null) opencv_core.cvReleaseData(Buffer); // si Buffer existe, on l'efface
		//if (Memory!=null) opencv_core.cvReleaseData(Memory); // si Memory existe, on l'efface
		//if (BufferGray!=null) opencv_core.cvReleaseData(BufferGray); // si BufferGray existe, on l'efface
		//if (Trans!=null) opencv_core.cvReleaseData(Trans); // si Trans existe, on l'efface

	if (buffers) { // si buffers
		if (Buffer!=null) opencv_core.cvReleaseImage(Buffer); // si Buffer existe, on l'efface
		
		if (BufferR!=null) opencv_core.cvReleaseImage(BufferR); // si BufferR existe, on l'efface
		if (BufferG!=null) opencv_core.cvReleaseImage(BufferG); // si BufferG existe, on l'efface
		if (BufferB!=null) opencv_core.cvReleaseImage(BufferB); // si BufferB existe, on l'efface
		
		//if (BufferA!=null) opencv_core.cvReleaseImage(BufferA); // si BufferA existe, on l'efface
		
		if (BufferGray!=null) opencv_core.cvReleaseImage(BufferGray); // si BufferGray existe, on l'efface
				
		//--- buffer 16S 3 canaux
		if (Trans16S3C!=null) opencv_core.cvReleaseImage(Trans16S3C); // si Trans existe, on l'efface
		if (Trans16S3C1!=null) opencv_core.cvReleaseImage(Trans16S3C1); // si Trans existe, on l'efface
		if (Trans16S3C2!=null) opencv_core.cvReleaseImage(Trans16S3C2); // si Trans existe, on l'efface

		//--- buffer 16S 1 canal
		if (Trans16S1C!=null) opencv_core.cvReleaseImage(Trans16S1C); // si Trans existe, on l'efface
		if (Trans16S1C1!=null) opencv_core.cvReleaseImage(Trans16S1C1); // si Trans existe, on l'efface
		if (Trans16S1C2!=null) opencv_core.cvReleaseImage(Trans16S1C2); // si Trans existe, on l'efface

		//--- buffer 8U 3 canaux
		if (Trans8U3C!=null) opencv_core.cvReleaseImage(Trans8U3C); // si Trans existe, on l'efface
		if (Trans8U3C1!=null) opencv_core.cvReleaseImage(Trans8U3C1); // si Trans existe, on l'efface
		if (Trans8U3C2!=null) opencv_core.cvReleaseImage(Trans8U3C2); // si Trans existe, on l'efface
		
		//--- buffer 8U 1 canal
		if (Trans8U1C!=null) opencv_core.cvReleaseImage(Trans8U1C); // si Trans existe, on l'efface
		if (Trans8U1C1!=null) opencv_core.cvReleaseImage(Trans8U1C1); // si Trans existe, on l'efface
		if (Trans8U1C2!=null) opencv_core.cvReleaseImage(Trans8U1C2); // si Trans existe, on l'efface

		//--- buffer 32F 3 canaux
		if (Trans32F3C!=null) opencv_core.cvReleaseImage(Trans32F3C); // si Trans existe, on l'efface

	} // -- fin if buffers 
	
		if (memory) { 
			
			if (Memory!=null) opencv_core.cvReleaseImage(Memory); // si Memory existe, on l'efface
			if (Memory2!=null) opencv_core.cvReleaseImage(Memory2); // si Memory existe, on l'efface
			
			//--- buffer 8U 1 canal
			if (Trans8U1CMemory!=null) opencv_core.cvReleaseImage(Trans8U1CMemory); // si Trans existe, on l'efface
			if (Trans8U1C1Memory!=null) opencv_core.cvReleaseImage(Trans8U1C1Memory); // si Trans existe, on l'efface
			if (Trans8U1C2Memory!=null) opencv_core.cvReleaseImage(Trans8U1C2Memory); // si Trans existe, on l'efface

			
		} // fin if memory 
		
		//---- Buffer publics --- 
		
		//--- définit objet mySize ---
		opencv_core.CvSize mySize=opencv_core.cvSize(widthIn,heightIn); // objet taille de l'image

		if (buffers) {
			
		// crée une image IplImage 8bits, 3 canaux (rgb)
		Buffer = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 3); // 3 canaux - RGB -- 

		//--- crée 3 buffers 1 canal 8 bits non signés = 1 canal par couleur 
		BufferR = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 1); // 1 canal - canal rouge
		BufferG = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 1); // 1 canal - canal vert 
		BufferB = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 1); // 1 canal - canal bleu 

		//BufferA = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 1); // 1 canal - canal transparence 

		//--- crée un buffer 1 canal 8 bits non signés = niveaux de gris 
		BufferGray = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 1); // 1 canal - niveaux de gris 
	
		
		//----- Buffers static de travail --- 
		
		//-- 16S 3C ---
		
		// crée une image Trans de travail, 3 canaux 16S
		Trans16S3C = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_16S, 3); //crée image Ipl 16bits Signés - 3 canaux

		// crée une image Trans de travail, 3 canaux 16S
		Trans16S3C1 = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_16S, 3); //crée image Ipl 16bits Signés - 3 canaux

		// crée une image Trans de travail, 3 canaux 16S
		Trans16S3C2 = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_16S, 3); //crée image Ipl 16bits Signés - 3 canaux

		//-- 16S 1C ---
		
		// crée une image Trans de travail, 1 canal 16S
		Trans16S1C = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_16S, 1); //crée image Ipl 16bits Signés - 1 canal

		// crée une image Trans de travail, 1 canal 16S
		Trans16S1C1 = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_16S, 1); //crée image Ipl 16bits Signés - 1 canal

		// crée une image Trans de travail, 1 canal 16S
		Trans16S1C2 = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_16S, 1); //crée image Ipl 16bits Signés - 1 canal
		
		//-- 8U - 3C ---
		
		// crée une image Trans de travail, 3 canaux 8U
		Trans8U3C = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 3); //crée image Ipl 8bits non Signés - 3 canaux

		// crée une image Trans de travail, 3 canaux 8U
		Trans8U3C1 = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 3); //crée image Ipl 8bits non Signés - 3 canaux

		// crée une image Trans de travail, 3 canaux 8U
		Trans8U3C2 = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 3); //crée image Ipl 8bits non Signés - 3 canaux

		//-- 8U - 1C ---

		// crée une image Trans de travail, 1 canal 8U
		Trans8U1C = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 1); //crée image Ipl 8bits non Signés - 1 canal

		// crée une image Trans de travail, 1 canal 8U
		Trans8U1C1 = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 1); //crée image Ipl 8bits non Signés - 1 canal

		// crée une image Trans de travail, 1 canal 8U
		Trans8U1C2 = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 1); //crée image Ipl 8bits non Signés - 1 canal

		//-- 32F - 1C ---
		
		// crée une image Trans de travail, 3 canaux 32F
		Trans32F3C = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_32F, 3); //crée image Ipl 32F (à virgule - signés) - 3 canaux

		} // fin Buffer
		
		if (memory) { // if memory 
			
			//opencv_core.CvSize mySize=opencv_core.cvSize(widthIn,heightIn); // objet taille de l'image
			Memory = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 3); // 3 canaux - RGB -- 

			//opencv_core.CvSize mySize=opencv_core.cvSize(widthIn,heightIn); // objet taille de l'image
			Memory2 = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 3); // 3 canaux - RGB -- 

			//-- 8U - 1C ---

			// crée une image Trans de travail, 1 canal 8U
			Trans8U1CMemory = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 1); //crée image Ipl 8bits non Signés - 1 canal

			// crée une image Trans de travail, 1 canal 8U
			Trans8U1C1Memory = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 1); //crée image Ipl 8bits non Signés - 1 canal

			// crée une image Trans de travail, 1 canal 8U
			Trans8U1C2Memory = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 1); //crée image Ipl 8bits non Signés - 1 canal

		} //fin if memory 
		
		
	} // -------- fin allocate ---------- 
	
	//--------------------- fonction allocate Buffer et memory ---------------------- 

	public void allocate (int widthIn, int heightIn) {
		
		allocate (widthIn, heightIn, true, true); 
		
	}
	//--------------------- fonction allocate Buffer seul  ---------------------- 

	public void allocateBuffer (int widthIn, int heightIn) {
		
		allocate (widthIn, heightIn, true, false); 
		
	}

	
	//--------------------- fonction allocate memory seul ---------------------- 
	public void allocateMemory (int widthIn, int heightIn) {
		
		allocate (widthIn, heightIn, false, true); 
		
	}

	//--------------------- fonction allocate memory seul ---------------------- 
	public void allocateMemory2 (int widthIn, int heightIn) {
	
		
		//--- définit objet mySize ---
		opencv_core.CvSize mySize=opencv_core.cvSize(widthIn,heightIn); // objet taille de l'image

		if (Memory2!=null) opencv_core.cvReleaseImage(Memory2); // si Memory2 existe, on l'efface
		
		//opencv_core.CvSize mySize=opencv_core.cvSize(widthIn,heightIn); // objet taille de l'image
		Memory2 = opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 3); // 3 canaux - RGB -- 

		
	}

	
	//=== fonction heigt() : renvoie le height du/des buffers opencv ===
	public int height() {
		
		return (Buffer.height()); 
	}
	
	//=== fonction width() : renvoie le width du/des buffers opencv ===
	public int width() {
		
		return (Buffer.width()); 
	}

	//=== fonction area() : renvoie l'aire du /des buffers opencv ===
	public long area() {
		
		return (Buffer.height()*Buffer.width()); 
	}

	
	//==== fonction copy(PImage) : copie le PImage dans le buffer ===
	public void copy (PImage imgIn) {
		
		Buffer=this.fromPImage(imgIn, Buffer); // copie le PImage dans le buffer IplImage 
		
	}

	//==== fonction copy(IplImage) : copie le IplImage dans le buffer ===
	public void copy (opencv_core.IplImage iplImgIn) {
		
		opencv_core.cvCopy(iplImgIn, Buffer); // copie l'image Ipl en entrée dans le Buffer principal
		
	}

	//==== fonction getBuffer : récupère le buffer dans un PImage ===
	public PImage getBuffer () {
		
		//PImage imgOut; 
		
		//imgOut=this.toPImage(Buffer); // copie dans le buffer IplImage dans le PImage
		
		return(this.toPImage(Buffer));
		
	}

	//==== fonction image : récupère le buffer dans un PImage ===
	public PImage image () {
		
		
		return(this.toPImage(Buffer));
		
	}

	
	//==== fonction copyToGray(PImage) : copie le PImage dans le buffer ===
	//-- idem copy --
	public void copyToGray (PImage imgIn) {
		
		PImage imgOut=new PImage(); // crée une nouvelle instance pour éviter passage référence seule du PImage reçu
		
		imgOut=imgIn; 
		
		this.copyTo(imgOut, Trans8U3C ); // pour ne pas modifier le buffer principal 
		
		//opencv_core.cvCopy(Trans8U3C ,Buffer); // copie l'image - debug 
		
		this.gray(Trans8U3C); // la fonction gray copie l'image dans le buffer gray
		
		
		
	} // fin copyToGray pour PImage

	//==== fonction copyToGray() : copie buffer principal dans buffer Gray ===
	//-- idem copy --
	public void copyToGray () {
		
		//opencv_imgproc.cvCvtColor(Buffer, BufferGray, opencv_imgproc.CV_RGB2GRAY); // bascule en niveaux de gris 
		this.copyToGray(Buffer);
		
	} // fin copyToGray pour buffer principal 

	//==== fonction copyToGray(IplImage) : copie IplImage dans le buffer ===
	//-- idem copy --
	public void copyToGray (opencv_core.IplImage iplImgIn) {
		
		if (iplImgIn.nChannels()==1){ // si l'image Ipl a bien 1 seul canal = est en niveau de gris 
		
			opencv_core.cvCopy(iplImgIn, BufferGray); // copie l'image Ipl en entrée dans le Buffer Gray
			
		}  // fin if
		
		if (iplImgIn.nChannels()==3){ // si l'image Ipl a bien 1 seul canal = est en niveau de gris 
			
			opencv_imgproc.cvCvtColor(iplImgIn, BufferGray, opencv_imgproc.CV_RGB2GRAY); // bascule en niveaux de gris 
			
		}  // fin if
			
		}// fin copyToGray pour IplImage

	//======= fonction copyTo(PImage src, IplImage dest)
	public void copyTo(PImage imgSrc, opencv_core.IplImage iplImgDest) {
	
		PImage imgOut=new PImage(); // crée une nouvelle instance pour éviter passage référence seule du PImage reçu
		
		imgOut=imgSrc; 

		iplImgDest=this.fromPImage(imgOut, iplImgDest); // copie dans le IplImage le PImage
		
	}

	//======= fonction copyToMemory(PImage src)
	public void copyToMemory(PImage imgIn) {
		
		remember(imgIn);
		
	}

	
	//======= fonction copyTo(IplImage src, IplImage dest)
	public void copyTo(opencv_core.IplImage iplImgSrc, opencv_core.IplImage iplImgDest) {
		
		if (iplImgSrc.nChannels()==iplImgDest.nChannels()){ // si IplImage source et destination ont le même nombre de canaux
			
			opencv_core.cvCopy(iplImgSrc, iplImgDest); // copie l'image Ipl en entrée dans le IplImage destination
			
		}  // fin if
		
		if ((iplImgSrc.nChannels()==1)&&(iplImgDest.nChannels()==3)){ // si IplImage source 1 canal et destination 3 canaux
			
			opencv_imgproc.cvCvtColor(iplImgSrc, iplImgDest, opencv_imgproc.CV_GRAY2RGB); // bascule en RGB 
			
		}  // fin if
		
		
		
	}
	
	//==== fonction getBufferGray : récupère le buffer Gray dans un PImage ===
	public PImage getBufferGray() {
		
		//PImage imgOut; 
		
		//imgOut=this.toPImage(Buffer); // copie dans le buffer IplImage dans le PImage
		
		return(this.toPImage(BufferGray));
	
	}
	
	//--------------- fonctions buffer RGB -------------- 
	
	//--- getBufferR
	public PImage getBufferR() {
		
		opencv_core.cvSet(Trans8U3C, opencv_core.cvScalarAll(0)); // vide le trans
		opencv_core.cvMerge(null, null, BufferR, null, Trans8U3C); // transfert le canal rouge dans Trans
		return(this.toPImage(Trans8U3C));
	}


	//--- getBufferG
	public PImage getBufferG() {
		
		opencv_core.cvSet(Trans8U3C, opencv_core.cvScalarAll(0)); // vide le trans
		opencv_core.cvMerge(null, BufferG, null, null, Trans8U3C); // transfert le canal vert dans Trans
		return(this.toPImage(Trans8U3C));
	}

	//--- getBufferB
	public PImage getBufferB() {
		
		opencv_core.cvSet(Trans8U3C, opencv_core.cvScalarAll(0)); // vide le trans
		opencv_core.cvMerge(BufferB, null, null, null, Trans8U3C); // transfert le canal bleu dans Trans
		return(this.toPImage(Trans8U3C));
	}

	//--------------- fonctions Memory ------------------- 
	
	//==== fonction remember(PImage) : copie le PImage dans le memory ===
	public void remember (PImage imgIn) {
		
		PImage imgOut=new PImage(); // crée une nouvelle instance pour éviter passage référence seule du PImage reçu
		
		imgOut=imgIn; 
		
		Memory=this.fromPImage(imgOut, Memory); // copie le PImage dans le memory  IplImage 
		
		
	} // fin remember


	//==== fonction remember() : copie le buffer IplImage dans le memory IplImage ===

	public void remember() {
		
			opencv_core.cvCopy(Buffer, Memory); // copie l'image Ipl Buffer en entrée dans le memory
			
	
			
		}// fin rememberBuffer

	//==== fonction restore() : copie le Memory IplImage dans le buffer principal IplImage ===

	public void restore() {
		
			copyTo(Memory, Buffer); // copie l'image Ipl Memory en entrée dans le Buffer principal
			
	
			
		}// fin restore
	
	//==== fonction restore2() : copie le Memory IplImage dans le buffer principal IplImage ===

	public void restore2() {
		
			copyTo(Memory2, Buffer); // copie l'image Ipl Memory en entrée dans le Buffer principal
			
	
			
		}// fin restore2

	//==== fonction getMemory : récupère le Memory dans un PImage ===
	public PImage getMemory () {
		
		//PImage imgOut; 
		
		//imgOut=this.toPImage(Buffer); // copie dans le buffer IplImage dans le PImage
		
		return(this.toPImage(Memory));
		
	} // fin getMemory
	
	//--------------- fonctions Memory2 ------------------- 
	
	//==== fonction remember2(PImage) : copie le PImage dans le memory2 ===
	public void remember2 (PImage imgIn) {
		
		PImage imgOut=new PImage(); // crée une nouvelle instance pour éviter passage référence seule du PImage reçu
		
		imgOut=imgIn; 
		
		Memory2=this.fromPImage(imgOut, Memory2); // copie dans le memory2  IplImage le PImage
		
	} // fin remember

	//==== fonction remember2() : copie le buffer IplImage dans le memory2 IplImage ===

	public void remember2 () {
		
			opencv_core.cvCopy(Buffer, Memory2); // copie l'image Ipl Buffer en entrée dans le memory2
			
	
			
		}// fin remember2Buffer
	

	//==== fonction getMemory2 : récupère le Memory2 dans un PImage ===
	public PImage getMemory2 () {
		
		//PImage imgOut; 
		
		//imgOut=this.toPImage(Buffer); // copie dans le buffer IplImage dans le PImage
		
		return(this.toPImage(Memory2));
		
	} // fin getMemory2
	
	//------- fonctions lecture fichier image ------------------
	
	//==== fonction loadImage : charge un fichier image dans le Buffer ===
	
	public void loadImage (String cheminFichierIn) {
	
	
		// charge le fichier et l'adapte à la taille du fichier destination
		opencv_imgproc.cvResize(opencv_highgui.cvLoadImage(cheminFichierIn),Buffer);
		
		
		
	} // fin loadImage

	
	//=== fonction loadImage avec taille  : redimensionne les buffer et charge un fichier image dans le Buffer 
	
	public void loadImage (String cheminFichierIn, int widthIn, int heightIn) {
		
		allocate(widthIn, heightIn,true,false); // reconfigure la taille des Buffers sans initialiser le Memory... 
		
		// charge le fichier et l'adapte à la taille du fichier destination
		opencv_imgproc.cvResize(opencv_highgui.cvLoadImage(cheminFichierIn),Buffer);
		
		
	} // fin loadImage

	
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////   OPERATIONS DE BASE SUR MATRICES OPENCV    ////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	//--- fonction fill : remplit tous les pixels d'une Image / d'un canal avec une même valeur 
	public void fill (opencv_core.IplImage iplImgIn, int valRIn, int valGIn, int valBIn) {
		
		//La fonction remplit tous les pixels d'un IplImage avec une même valeur pour chaque canal
		
		if (iplImgIn.nChannels()==3) { // si 3 canaux 
		
		opencv_core.CvScalar myRGB =opencv_core.CV_RGB(valRIn,valGIn,valBIn); // crée un scalaire de 3 valeurs RGB... Equiv color RGB Processing
		opencv_core.cvSet(iplImgIn, myRGB); // rempli tous les pixels de l'image  avec les valeurs du scalaire
		
		//alternative possible  
		//opencv_core.cvSet(iplImgIn, opencv_core.CV_RGB(valRIn,valGIn,valBIn)); // rempli tous les pixels de l'image  avec les valeurs du scalaire
		
		}

		if (iplImgIn.nChannels()==1) { // si 1 canal 
			
			opencv_core.CvScalar myScalar =opencv_core.cvScalarAll(valRIn); // crée un scalaire ... 
			opencv_core.cvSet(iplImgIn, myScalar); // rempli tous les pixels de l'image  avec les valeurs du scalaire
			
			//alternative possible  
			//opencv_core.cvSet(iplImgIn, opencv_core.CV_RGB(valRIn,valGIn,valBIn)); // rempli tous les pixels de l'image  avec les valeurs du scalaire
			
			}

	}
	
	//--- variantes fill
	public void fill (opencv_core.IplImage iplImgIn, int val) { // Ipl Image et 1 valeur 
		
		fill(iplImgIn, val,val,val); // utilise la même valeur pour les 3 canaux 
		
	}

public void fill (int valRIn, int valGIn, int valBIn) { // applique au Buffer 3 valeurs 
		
		fill(Buffer, valRIn,valGIn,valBIn);
		
	}

	public void fill (int val) { // applique au Buffer 1 valeur 
		
		fill(Buffer, val,val,val);
		
	}

	public void fill (boolean val) { // remplit Buffer à partir  1 valeur booleenne 
		
		// --- blanc si true, noir si false -- 
		if (val) fill(Buffer, 255,255,255); else fill(Buffer, 0,0,0);  
		
	}

	//// fonction addition pondérée de 2 images /////
	public void addWeighted (opencv_core.IplImage iplImgSrc1In, float coeff1In, opencv_core.IplImage iplImgSrc2In,  float coeff2In, float delta, opencv_core.IplImage iplImgDest) {
		
		opencv_core.cvAddWeighted(iplImgSrc1In, coeff1In, iplImgSrc2In, coeff2In, delta, iplImgDest);
		
	}
	
	/// fonction moyenne entre 2 images  ///
	public void average (opencv_core.IplImage iplImgSrc1In, opencv_core.IplImage iplImgSrc2In, opencv_core.IplImage iplImgDest) {
		
		//---- ajoute les 2 images affectée d'un coeff 0.5 
		opencv_core.cvAddWeighted(iplImgSrc1In, (float)0.5, iplImgSrc2In, (float)0.5,(float)0, iplImgDest);
		
	}
	
	//// Fonctions d'accumulation d'image ///
	
	//public void accumulate(opencv_core.IplImage iplImgSrcIn, opencv_core.IplImage iplImgAccIn, float coeff, boolean absDiffIn, float seuilDetectIn, float seuilRAZIn, boolean debugIn){
	public void accumulate(opencv_core.IplImage iplImgSrcIn, opencv_core.IplImage iplImgAccIn, float coeff, opencv_core.IplImage iplImgAbsDiffIn, float seuilDetectIn, float seuilRAZIn, boolean debugIn){	
		//--- la fonction accumulate réalise une accumulation par somme pondérée d'une image
		
		//--- la fonction reçoit
		// une image source correspondant au frame courant du flux vidéo
		// une image accumulateur destination correspondant au fond mémorisé
		
		// Une image correspondant au résultat de la différence absolue entre l'image courante et le fond mémorisé
		
		// un coeff fixant la pondération du frame à ajouter par rapport au fond mémorisé - ceci fixe aussi la vitesse de correction
		// une valeur seuil de détection au delà de laquelle on ne réalise pas l'accumulation = présente objet significatif - val * 100 000 
		// une valeur seuil de RAZ pour laquelle on met l'image à 0 si la valeur est suffisamment basse - val * 100 000 
		
		//if (debugIn) PApplet.println("Somme Img Src In = " + sum(iplImgSrcIn)); 
			
		if (sum(iplImgAbsDiffIn)<(seuilDetectIn*100000)){ // si la valeur détection objet de l'image en entrée n'est pas dépassée
			// on réalise l'addition du frame à l'accumulateur
			
			if (debugIn) PApplet.println("Seuil détect pas dépassé "); 
			
			opencv_core.cvConvert(iplImgAccIn,Trans32F3C); // passage par image 32F obligatoire pour accumulateur
		
			opencv_imgproc.cvRunningAvg(iplImgSrcIn, Trans32F3C, coeff, null); 
		
			opencv_core.cvConvert(Trans32F3C,iplImgAccIn); // récupère le 32F 
		
			if (sum(iplImgAbsDiffIn)<(seuilRAZIn*100000)){ // si la valeur seuil minimal est atteinte
	
				if (debugIn) PApplet.println("Inférieur Seuil RAZ "); 

				fill(iplImgAbsDiffIn, 0); // on rempli tous les pixels de la différence à 0 
				
			} // fin si valeur seuil minimale atteinte
		
		} // fin si comparaison seuilDetect
		
		
		
	} // fin fonction accumulate()
	
	
	//// ------------- Fonction sum() : renvoie la somme de la valeur des Pixels d'une image
	
	public float sum(opencv_core.IplImage iplImgSrc) {
		
		double sumOut=0; 
		
		opencv_core.CvScalar sumScalar=opencv_core.cvSum(iplImgSrc);
		
		sumOut=sumScalar.red()+sumScalar.green()+ sumScalar.blue(); // additionne les 3 valeurs 
		
		return ((float)sumOut); // renvoie la valeur 
		
	}
	
	//-- forme minimale
	public float sum() {
		
		return(sum(Buffer)); 
		
	}

	//// Fonction sumR() : renvoie la somme de la valeur des Pixels du canal Rouge d'une image
	
	public float sumR(opencv_core.IplImage iplImgSrc) {
		
		float sumOutR=0; 
		
		opencv_core.CvScalar sumScalar=opencv_core.cvSum(iplImgSrc);
		
		sumOutR=(float)sumScalar.red(); // renvoie la somme des pixels du canal rouge
		
		return (sumOutR); // renvoie la valeur 
		
	}

	//-- forme minimale
	public float sumR() {
		
		return(sumR(Buffer)); 
		
	}

	//// Fonction sumG() : renvoie la somme de la valeur des Pixels du canal Vert d'une image
	
	public float sumG(opencv_core.IplImage iplImgSrc) {
		
		float sumOutG=0; 
		
		opencv_core.CvScalar sumScalar=opencv_core.cvSum(iplImgSrc);
		
		sumOutG=(float)sumScalar.green(); // renvoie la somme des pixels du canal rouge
		
		return (sumOutG); // renvoie la valeur 
		
	}

	//-- forme minimale
	public float sumG() {
		
		return(sumG(Buffer)); 
		
	}

	//// Fonction sumB() : renvoie la somme de la valeur des Pixels du canal Bleu d'une image
	
	public float sumB(opencv_core.IplImage iplImgSrc) {
		
		float sumOutB=0; 
		
		opencv_core.CvScalar sumScalar=opencv_core.cvSum(iplImgSrc);
		
		sumOutB=(float)sumScalar.blue(); // renvoie la somme des pixels du canal rouge
		
		return (sumOutB); // renvoie la valeur 
		
	}

	//-- forme minimale
	public float sumB() {
		
		return(sumB(Buffer)); 
		
	}
	
	//// Fonction sumRGB() : renvoie la somme de la valeur des Pixels d'une image par canal dans un tableau 3 valeurs
	// canal R avec indice 0, la canal G avec indice 1, et canal bleu avec indice 2
	
	public float[] sumRGB(opencv_core.IplImage iplImgSrc) {
		
		float[] sumOut=new float[3]; 
		
		opencv_core.CvScalar sumScalar=opencv_core.cvSum(iplImgSrc);
		
		sumOut[0]=(float)sumScalar.red(); // renvoie la valeur du canal rouge
		sumOut[1]=(float)sumScalar.green(); // renvoie la valeur du canal vert
		sumOut[2]=(float)sumScalar.blue(); // renvoie la valeur du canal bleu
		
		
		return (sumOut); // renvoie la valeur 
		
	}

	//-- forme minimale
	public float[] sumRGB() {
		
		return(sumRGB(Buffer)); 
		
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////   METHODES DE TRAITEMENT D'IMAGE OPENCV    ////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/////////////////////////////////////  Gray ///////////////////////////////////////////////////////////////////
	
	//======= fonction gray() : transforme une IplImage RGB en niveaux de gris ===

	public opencv_core.IplImage gray (opencv_core.IplImage iplImgIn) { // la fonction reçoit le IplImage à traiter et  renvoie le IplImage modifié

		opencv_imgproc.cvCvtColor(iplImgIn, BufferGray, opencv_imgproc.CV_RGB2GRAY); // bascule en niveaux de gris 

		opencv_imgproc.cvCvtColor(BufferGray, iplImgIn, opencv_imgproc.CV_GRAY2RGB); // rebascule en RGB 
		// les 3 canaux du buffer RGB sont identiques = l'image est en niveaux de gris 

		// la copie est conservée dans le buffer Gray

	  
	  return(iplImgIn); 
	  
	} // fin fonction gray principale 

	// fonction gray buffer
	public void gray () { // convertit le buffer couleur en niveaux de gris 
		
		gray(Buffer); // applique la fonction gray() au Buffer principal

	} // fin fonction gray Buffer 
	
	//---- fonction gray (string)
	public void gray (String stringIn) { // convertit le buffer couleur en niveaux de gris
				
		gray(selectBuffer(stringIn)); // applique fonction gray() sur le Buffer voulu 

	} // fin fonctino gray String
	
	
	/////////////////////// Invert ///////////////////////////////////////////////////////////
	
	
	//------------------ fonction INVERT principale : inverse une ipl image 8 bits - 3 canaux ---------- 
	public void invert(opencv_core.IplImage iplImgIn) {

		/*
		//---- objet IplImage utilisé par la fooncton --- 
		
		opencv_core.IplImage iplImgSrc=new opencv_core.IplImage(); 
		
		//------ gestion de l'image à utiliser --- 
		if (stringIn=="MEMORY2") iplImgSrc=Memory2;
		else if (stringIn=="MEMORY") iplImgSrc=Memory;		
		else iplImgSrc=Buffer;
*/
		
		/*
		//--- extraction taille image 
		opencv_core.CvSize  mySize=iplImgIn.cvSize(); // récupère la taille de l'image - Cvsize est un objet contenant 2 valeurs

		//---- création d'un masque tous pixels à 1 ---- 
		opencv_core.IplImage iplImgMask= opencv_core.cvCreateImage(mySize, opencv_core.IPL_DEPTH_8U, 1); // crée une image IplImage 8bits , 1 canal

		//opencv_core.cvSet(iplImgMask, opencv_core.CV_RGB(1,1,1)); // rempli tous les pixels de l'image  avec les valeurs du scalaire
		opencv_core.cvSet(iplImgMask, opencv_core.cvScalarAll(1)); // rempli tous les pixels de l'image  avec les valeurs du scalaire
		// tous les points du masque sont mis à 1
		*/
		
		// TODO : support image 1 canal
		
		//--- utilise plutôt image trans déjà déclarée = gain temps + évite surcharge mémoire
		opencv_core.IplImage iplImgMask=Trans8U1C; // renomme image trans
		
		opencv_core.cvSet(iplImgMask, opencv_core.cvScalarAll(1)); // rempli tous les pixels de l'image  avec les valeurs du scalaire
		// tous les points du masque sont mis à 1
		
		
		//--- soustraction pixel par pixel 255-valeur = inverse l'image 
		opencv_core.CvScalar myRGB = opencv_core.CV_RGB(255, 255, 255); // crée scalaire de 3 valeurs
		
		  // static void 	cvSub(opencv_core.CvArr src1, opencv_core.CvArr src2, opencv_core.CvArr dst, opencv_core.CvArr mask) : src1 - src2
		  //static void 	cvSubS(opencv_core.CvArr src, opencv_core.CvScalar value, opencv_core.CvArr dst, opencv_core.CvArr mask) : src1-src2
		  // static void 	cvSubRS(opencv_core.CvArr src, opencv_core.CvScalar value, opencv_core.CvArr dst, opencv_core.CvArr mask) : src2 - src1 (Reverse Substract)
		  
		  opencv_core.cvSubRS(iplImgIn, myRGB,iplImgIn,iplImgMask); // soustraction inverse du scalaire 
		  // opencv_core.cvXorS( iplImgSrc, opencv_core.cvScalarAll(255), iplImgDest, iplImgMask ); // équivalent... 
		
		  //opencv_core.cvReleaseImage(iplImgSrc); // libère mémoire utilisée par iplImgSrc 
		 
		
	} // fin invert principale

	//--- fonction invert String 
	public void invert(String stringIn) {
		
		invert(selectBuffer(stringIn)); // sélectionne le buffer à partir de la chaine de caractère reçue
		
	} // fin fonction Invert 


	//--- fonctin Invert sur Buffer
	public void invert() {
		
		invert(Buffer);
		
	} // fin fonction Invert 
	
	///////////////////////////////////// Brightness Contrast /////////////////////////////////////////////////////////
	
	
	//======== setBrightnessContrast : fixe le contraste et la luminosité =======

	public opencv_core.IplImage setBrightnessContrast(opencv_core.IplImage iplImgIn, int brightnessIn, int contrastIn) { 

		// la fonction reçoit l'objet IplImage source et les valeurs du contraste et de la luminsoté
		// contrast entre -128 et +127 
		// brightnesse entre -128 et + 127
	
		/* revoir--- 
		
		float alpha=p.constrain(contrastIn, (float)1.0, (float)3.0); // fixe les limites - utilise la fonction constrain de Processing
		float beta=p.constrain(brightnessIn,0,100); // fixe les limites -  utilise la fonction constrain de Processing
		
		
		// chaque pixel est fixé par (alpha* valeur) + beta
			
	  //---- créer un objet opencv CvScalar (conteneur de 1 à 4 valeur) pour contenir valeur r,g,b d'un pixel
	  opencv_core.CvScalar toPixel= new opencv_core.CvScalar(3); // scalaire contenant le groupe de valeur r,g,b d'un pixel x,y de l'image

	  //---- passer en revue et modifier les pixels de l'IplImage --- 

	  for (int x=0; x <IplImageIn.width(); x++) { //-- défile les pixels de l'image
	    for (int y=0; y <IplImageIn.height(); y++) {
	    
	    toPixel=opencv_core.cvGet2D(IplImageIn,y,x); // récupère la valeur du pixel - attention y,x et pas x,y !
	    //--- passer par cvGetCol pour accélérer ? 
	    
	    toPixel.red((toPixel.red()*alpha)+beta); // modifie  la valeur du canal du pixel 
	    toPixel.green((toPixel.green()*alpha)+beta); // modifie  la valeur du canal du pixel 
	    toPixel.blue((toPixel.blue()*alpha)+beta); // modifie  la valeur du canal du pixel 
	    	    
	    } // fin y 
	  } // fin x 

		
	  // met à jour les variables de classe en conséquence : 
	  contrast=alpha; 
	  brightness=beta; 
	  
	  */
		
		//-- on considère que l'image reçue est 8 bits - 3 canaux 
		
		BytePointer myPtr = new BytePointer(256*3); // crée un pointeur de la taille voulue = nombre valeurs x nombres canaux
		//BytePointer myPtr = new BytePointer(IplImgIn.depth()*IplImgIn.nChannels()); // crée un pointeur de la taille voulue = nombre valeurs x nombres canaux
		
		
		 opencv_core.CvMat matrix=  opencv_core.cvMat(1, 256, opencv_core.CV_8UC3, myPtr) ; // crée un CvMat avec taille de donnée unitaire 8 bits - 3 canaux
		 //-- on crée un CvMat 8 bits / 3 canaux   car le CvMat doit avoir le même nombre de canaux et le même nombre bit (depth) 
		 // que le  IplImage source utilisé avec la fonction cvLUT()
		 
		 //---- calcul des coefficients pour le contraste et la brillance
		 //--- pour chaque pixel, on aura : I= (a x I) + b
		 
		 double delta, a, b; // variables utiles 
		 
		 	if ( contrastIn>0 ) {
		 		
		          delta = (127*contrastIn) / 128;

		          a = 255 / ( 255-(delta*2) );
		          b = a * ( brightnessIn-delta);
	          
		        }

		    else {
		      
		       	delta = ((-128)*contrast) / 128;

			a = ( 256 -(delta*2) ) / 255.;
			b = ( a * brightnessIn ) + delta;

		        }
		 	
		 //--- met à jour le Cvmat avec les nouvelles valeurs calculées 
		 	 
		 	for( int i=0; i<256; i++ ) {
		   
				int value = PApplet.round( (float)((a*i)+b) ); // récupère valeur entière 
	
				//value=p.min(p.max(0,value),255); // équiv constrain 0-255
				value=PApplet.constrain(value,0,255); // équiv constrain 0-255
				
				opencv_core.cvSet1D(matrix, i, opencv_core.cvScalarAll(value)); // remplit les 3 canaux du CvMat à l'index voulu
		                
			}

		 //---- application du cvLUT en se basant sur le CvMat défini
		 	  
		    // static void 	cvLUT(opencv_core.CvArr src, opencv_core.CvArr dst, opencv_core.CvArr lut) 

		 	opencv_core.cvLUT( iplImgIn, iplImgIn, matrix );
		    
		    // cvLUT remplace dans l'image source chaque pixel par la valeur présente à chaque indice du tableau 1D CvMat utilisé
		    // le résultat est mis dans la destination 
		    // ceci limite les opérations au calcul de 255 valeurs au lieu de faire widthxheight calculs identiques

		 	//-- met à jour les variables static contrast et brightness
		 	contrast=contrastIn; 
		 	brightness=brightnessIn;

		//--- renvoie l'image modifiée --- 
	  
		return (iplImgIn);
		
}// fin fonction setBrightnessContrast
	
	//--- setBrightnessContrast buffer 
	public void setBrightnessContrast(int brightnessIn, int contrastIn) {
		
		setBrightnessContrast(Buffer, brightnessIn, contrastIn); // modifie contraste et luminosité du Buffer
		
	}
	
	//---- fonction contrast : applique contrast au buffer IplImage
	public void contrast(int contrastIn){
		
		Buffer=this.setBrightnessContrast(Buffer,brightness, contrastIn); // laisse brightness courant idem
	}

	//---- fonction contrast : applique contrast à un objet IplImage
	public void contrast(opencv_core.IplImage iplImgIn, int contrastIn){
		
		Buffer=this.setBrightnessContrast(iplImgIn,brightness, contrastIn); // laisse brightness courant idem
	}

	//---- fonction brightness : applique brightness au buffer IplImage
	public void brightness(int brightnessIn){
		
		Buffer=this.setBrightnessContrast(Buffer,brightnessIn, contrast); // laisse contrast courant idem
	}

	//---- fonction brightness : applique brightness à un objet IplImage
	public void brightness(opencv_core.IplImage iplImgIn, int brightnessIn){
		
		Buffer=this.setBrightnessContrast(iplImgIn,brightnessIn, contrast); // laisse contrast courant idem
	}

	///////////////////////////////////// extractRGB /////////////////////////////////////////////////////
	
	//--- fonction principale extractRGB() : extrait les 3 canaux RGB d'un IplImage et les mets dans les buffers RGB
	
	public void extractRGB(opencv_core.IplImage iplImgIn) {
		
		 opencv_core.cvSplit(iplImgIn, BufferB, BufferG, BufferR, null); // extrait les 3 canaux R G B : attention inversion RGB  

	}
	
	//--- fonction  extractRGB() : extrait les 3 canaux RGB du Buffer et les mets dans les buffers RGB
	
	public void extractRGB() {
		
		 //opencv_core.cvSplit(Buffer, BufferB, BufferG, BufferR, null); // extrait les 3 canaux R G B : attention inversion RGB
		extractRGB(Buffer); // appel fonction principale - préférable 

	}

	///////////////////////////////////// mergeRGB /////////////////////////////////////////////////////
	
	//--- fonction principale mergeRGB() : reconstruction IplImage destination à partir des canaux départ.. 
	
	public void mergeRGB(opencv_core.IplImage iplImgRIn, opencv_core.IplImage iplImgGIn, opencv_core.IplImage iplImgBIn,  opencv_core.IplImage iplImgDestIn) {
		
		// attention : les trois images R,G et B doivent être des mono-canal et Dest une triple canaux
		
		opencv_core.cvMerge(iplImgBIn,iplImgGIn,iplImgRIn, null, iplImgDestIn); // reconstruction IplImage destination à partir des canaux départ.. - attention inversion RGB
		
	}
	
	//--- fonction  mergeRGB par défaut : fusionne les 3 buffers R,G et B dans le buffer principal
	
	public void mergeRGB() {
		
		 
		mergeRGB(BufferR, BufferG, BufferB,Buffer); // appel fonction principale merge

	}

	//--- fonction  mergeRGB avec Dest seul  : fusionne les 3 buffers R,G et B dans l'image destination 
	
	public void mergeRGB(opencv_core.IplImage iplImgDestIn) {
		
		 
		mergeRGB(BufferR, BufferG, BufferB,iplImgDestIn); // appel fonction principale merge

	}

	///////////////////////////////////// Multiply : applique facteur multiplicateur aux canaux /////////////////////////////////////////////////////////
	
	public opencv_core.IplImage multiply (opencv_core.IplImage iplImgIn, double coeffRIn, double coeffGIn, double coeffBIn, boolean debugIn) {
		
		  //opencv_core.cvSet(Trans8U3C, opencv_core.CV_RGB(coeffRIn, coeffGIn, coeffBIn)); // remplit tous les pixels du IplImage avec le scalaire (coeffR, coeffG, coeffB)
		  // à revoir - passer par image en 32F pour supporter coeff float !!
		  
		  
		  //opencv_core.cvMul(iplImgIn, Trans8U3C, iplImgIn, 1); // multiplie les 2 images
		  
		  //--- conversion de l'image source en 16S 
		  //opencv_core.cvConvertScale(iplImgSrc, iplImgSrc16S, 256.0, -32768); // convertit 8U en 16S 
		  opencv_core.cvConvertScale(iplImgIn, Trans16S3C, 1, 0); // convertit 8U en 16S mais en gardant les valeurs 8U
	  
		  //---- application d'un coefficient à chaque canal ---- 
		  // calcul séparément pour chaque pixel R x coeff R , G x coeff G, B x coeff B
		  
		  //--- création de 3 matrices de float contenant les coeff RGB à appliquer :
		  opencv_core.cvSet(Trans32F3C, opencv_core.CV_RGB(coeffRIn, coeffGIn, coeffBIn)); // remplit tous les pixels du IplImage avec le scalaire (coeffB, coeffG, coeffR)
		  // pour pouvoir stocker des float, il faut que l'image soit une 32F !!
		  // attention coeff R et B PAS inversés  ici 
		  
		  if (debugIn) PApplet.println("Val(0,0) ="+opencv_core.cvGet2D(Trans32F3C, 0,0).val(0)); // lit la valeur... utilise fonction .val(i) du scalar renvoyé par cvGet2D
		  // val(0) renvoie canal Bleu, val(1) le canal vert et val(2) le canal Rouge

		  if (debugIn) PApplet.println("Val(0,0) ="+opencv_core.cvGet2D(Trans16S3C, 0,0).val(2)); // lit la valeur... utilise fonction .val(i) du scalar renvoyé par cvGet2D

				  
		  opencv_core.cvMul(Trans16S3C, Trans32F3C, Trans16S3C, 1); // multiplie les 2 images 
		  // ce qui réalise pour chaque pixel : R x coeff R , G x coeff G, B x coeff B
		  //-- nb ; pour la multiplication avec des float, il faut que l'image soit une 32F

		  //conversion d'un Objet Ipl16S en 8U mais sans changer la valeur
		  // opencv_core.cvConvertScale(iplImg16S,iplImg8U,(1.0/256.0),128);  conversion 16S vers 8U avec adaptation valeur pleine échelle 
		  opencv_core.cvConvertScale(Trans16S3C,Trans8U3C,1,0); // sans changer la valeur
		 
		  opencv_core.cvCopy(Trans8U3C, iplImgIn); // copie l'image Ipl en entrée dans le Buffer Gray
		
		return (iplImgIn);
	}
	
	//---- multiply(double coeff) sur buffer -- 
	
	public void multiply (double coeffIn) { // applique multiply() au buffer avec coeff commun 
		
		multiply(Buffer, coeffIn, coeffIn, coeffIn, false); // applique coeff différent pour les 3 canaux 
		
		
	}

//---- multiply(double coeff) sur iplImage -- 
	
	public void multiply (opencv_core.IplImage iplImgIn, double coeffIn) { // applique multiply() au buffer avec coeff commun 
		
		multiply(iplImgIn, coeffIn, coeffIn, coeffIn, false); // applique coeff différent pour les 3 canaux 
		
		
	}

	//---- multiply(double coeffRIn, double coeffGIn, double coeffBIn) sur buffer -- 
	
	public void multiply (double coeffRIn, double coeffGIn, double coeffBIn) { // applique multiply() au buffer avec coeff commun 
		
		multiply(Buffer, coeffRIn, coeffGIn, coeffBIn, false); // applique le même coeff pour les 3 canaux 
		
	}

	////////////////////////////////////// Threshold : applique un seuillage ////////////////////////////////////////

	//---- threshold : fonction principale 
	public opencv_core.IplImage threshold (opencv_core.IplImage iplImgIn, int seuilIn, String methodIn ) {
		
		gray(iplImgIn); // bascule en niveau de gris (dans tous les cas - le plus simple)
		
		
		  //static double cvThreshold(opencv_core.CvArr src, opencv_core.CvArr dst, double threshold, double max_value, int threshold_type) 
		  // méthode disponibles : CV_THRESH_BINARY, CV_THRESH_BINARY_INV, CV_THRESH_TRUNC, CV_THRESH_TOZERO, CV_THRESH_TOZERO_INV 

		//-- sélection de la méthode en fonction du String reçu
		if (methodIn=="BINARY")opencv_imgproc.cvThreshold(BufferGray, BufferGray, seuilIn, 255, opencv_imgproc.CV_THRESH_BINARY); // applique seuilllage
		if (methodIn=="BINARY_INV")opencv_imgproc.cvThreshold(BufferGray, BufferGray, seuilIn, 255, opencv_imgproc.CV_THRESH_BINARY_INV); // applique seuilllage
		if (methodIn=="TRUNC")opencv_imgproc.cvThreshold(BufferGray, BufferGray, seuilIn, 255, opencv_imgproc.CV_THRESH_TRUNC); // applique seuilllage
		if (methodIn=="TOZERO")opencv_imgproc.cvThreshold(BufferGray, BufferGray, seuilIn, 255, opencv_imgproc.CV_THRESH_TOZERO); // applique seuilllage
		if (methodIn=="TOZERO_INV")opencv_imgproc.cvThreshold(BufferGray, BufferGray, seuilIn, 255, opencv_imgproc.CV_THRESH_TOZERO_INV); // applique seuilllage

		opencv_imgproc.cvCvtColor(BufferGray, iplImgIn, opencv_imgproc.CV_GRAY2RGB); // rebascule en RGB
		
		return (iplImgIn);
	
	} // fin threshold

	//-- variante avec float --
	public void threshold (opencv_core.IplImage iplImgIn, float seuilIn, String methodIn ) {

			int seuilInt=(int)(255*seuilIn);
			threshold(iplImgIn, seuilInt, methodIn); 
	}
	
	//---- threshold : sur buffer avec paramètres seuil et méthode 	
	public void threshold (int seuilIn, String methodIn ) {

			threshold (Buffer, seuilIn, methodIn); 
		
	} //fin threshold

	//---- threshold : sur buffer avec paramètres seuil float et méthode 	
	public void threshold (float seuilIn, String methodIn ) {

		int seuilInt=(int)(255*seuilIn);
		threshold (Buffer, seuilInt, methodIn); 
		
	} //fin threshold

	//---- threshold : sur buffer avec seuill ---
	public void threshold (int seuilIn ) {

		threshold (Buffer, seuilIn, "BINARY"); 
	
	} // fin threshold

	//---- threshold : sur buffer avec seuil float---
	public void threshold (float seuilIn ) {

		int seuilInt=(int)(255*seuilIn);
		threshold (Buffer, seuilInt, "BINARY"); 
	
	} // fin threshold
	
	///////////////////////////////////// Flip : renverse/retourne image //////////////////////////////////////////////
	
	//-- flip principale : applique flip à Pimage
	public void flip(opencv_core.IplImage iplImgIn, String flipModeIn) {
		
		//--- la fonction peut recevoir les champs prédéfinis VERTICAL, HORIZONTAL et BOTH (sans " ")
		
		// renverse l'image selon mode voulu : 0 = vertical, 1 = horizontal, -1 = les 2
		if (flipModeIn=="VERTICAL") opencv_core.cvFlip(iplImgIn, iplImgIn, 0); // flip vertical 
		if (flipModeIn=="HORIZONTAL") opencv_core.cvFlip(iplImgIn, iplImgIn, 1); // flip horizontal 
		if (flipModeIn=="BOTH") opencv_core.cvFlip(iplImgIn, iplImgIn, -1); // flip vertical et horizontal 
		 
	} // fin flip 

	//--- applique flip au Buffer
	public void flip(String flipModeIn) {

		flip(Buffer,flipModeIn); 
	}
	
	/////////////////////////////////////   smooth  (obsolete : cf doc 2.3.1 ) /////////////////////////////////////////

	//======= fonction smooth() : applique un flou à une IplImage ===

	public opencv_core.IplImage smooth (opencv_core.IplImage iplImgIn) { // la fonction reçoit le IplImage à traiter et  renvoie le IplImage modifié

	  // javacv : static void 	opencv_imgproc.cvSmooth(opencv_core.CvArr src, opencv_core.CvArr dst, int smoothtype, int size1) 
	  // avec (doc opencv) : 
	  //
	  opencv_imgproc.cvSmooth(iplImgIn, iplImgIn, opencv_imgproc.CV_GAUSSIAN, 3); // applique un effet Flou gaussien - kernel 3x3
	  
	  return(iplImgIn); 
	}

	//--- Smooth avec paramètre ksize -------------
	public opencv_core.IplImage smooth (opencv_core.IplImage iplImgIn, int ksizeIn) { // la fonction reçoit le IplImage à traiter et taille kernel - renvoie le IplImage modifié

	  // javacv : static void 	opencv_imgproc.cvSmooth(opencv_core.CvArr src, opencv_core.CvArr dst, int smoothtype, int size1) 
	  opencv_imgproc.cvSmooth(iplImgIn, iplImgIn, opencv_imgproc.CV_GAUSSIAN, ksizeIn); // applique un effet Flou gaussien - kernel 3x3
	  
	  return(iplImgIn); 
	}

	//---- fonction smooth : applique smooth au buffer IplImage
	public void smooth(){
		
		Buffer=this.smooth(Buffer);
	}

	//---- fonction smooth : applique smooth au buffer IplImage avec taille kernel
	public void smooth(int ksizeIn){
		
		Buffer=this.smooth(Buffer, ksizeIn);
	}

	/////////////////////////////////////   Blur   ///////////////////////////////////////////////////////////////

	//======= fonction blur() : applique un flou à une IplImage ===

	public opencv_core.IplImage blur (opencv_core.IplImage iplImgIn) { // la fonction reçoit le IplImage à traiter et  renvoie le IplImage modifié

	  // javacv :  static void 	blur(opencv_core.CvArr src, opencv_core.CvArr dst, opencv_core.CvSize ksize, opencv_core.CvPoint anchor, int borderType) 
	  // avec (doc opencv) : 
	  //

		// application d'un flou avec noyau 3x3
		opencv_imgproc.blur( iplImgIn, iplImgIn, opencv_core.cvSize(3,3), opencv_core.cvPoint(-1,-1), opencv_imgproc.BORDER_DEFAULT); 
	  
	  return(iplImgIn); 
	}

	//--- smooth avec paramètre ksize -------------
	public opencv_core.IplImage blur (opencv_core.IplImage iplImgIn, int ksizeIn) { // la fonction reçoit le IplImage à traiter et taille kernel - renvoie le IplImage modifié

		  // javacv :  static void 	blur(opencv_core.CvArr src, opencv_core.CvArr dst, opencv_core.CvSize ksize, opencv_core.CvPoint anchor, int borderType) 

			// application d'un flou avec noyau 3x3
			opencv_imgproc.blur( iplImgIn, iplImgIn, opencv_core.cvSize(ksizeIn,ksizeIn), opencv_core.cvPoint(-1,-1), opencv_imgproc.BORDER_DEFAULT); 
	  
	  return(iplImgIn); 
	}

	//---- fonction blur: applique blur (flou) au buffer IplImage
	public void blur(){ //-- idem.. 
		
		Buffer=this.blur(Buffer);
	}


	//---- fonction blur: applique blur (flou) au buffer IplImage
	public void blur(int ksizeIn){ //-- idem.. 
		
		Buffer=this.blur(Buffer, ksizeIn);

	}
	
	
	////////////////////////////////////////////////// MIXEUR DE CANAUX //////////////////////////////////////////////////////////////
	
	//--- Fonction principale Mixeur de Canaux 
	//-- durée de l'ordre de 3 ms ... 
	public opencv_core.IplImage mixerRGB (opencv_core.IplImage iplImgIn, float coeffRIn, float coeffGIn, float coeffBIn, int canalOut, boolean grayOut, boolean debugIn) {
		
		// la fonction mixerRGB implémente un algorithme du logiciel The Gimp appelé "mixeur de canaux"
		// un des canaux RGB est choisi comme canal de sortie (R par défaut)
		// chaque pixel est recalculé en intégrant une certaine proportion des autres canaux selon : 
		// R = (R x coeff R) + (G x coeff G) + (B x coeff B)
		
		  //--- conversion de l'image source en 16S 
		  //opencv_core.cvConvertScale(iplImgSrc, iplImgSrc16S, 256.0, -32768); // convertit 8U en 16S 
		  opencv_core.cvConvertScale(iplImgIn, Trans16S3C, 1, 0); // convertit 8U en 16S mais en gardant les valeurs 8U
	  
		  //---- application d'un coefficient à chaque canal ---- 
		  // calcul séparément pour chaque pixel R x coeff R , G x coeff G, B x coeff B
		  
		  //--- création de 3 matrices de float contenant les coeff RGB à appliquer :
		  opencv_core.cvSet(Trans32F3C, opencv_core.CV_RGB(coeffRIn, coeffGIn, coeffBIn)); // remplit tous les pixels du IplImage avec le scalaire (coeffB, coeffG, coeffR)
		  // pour pouvoir stocker des float, il faut que l'image soit une 32F !!
		  // attention coeff R et B PAS inversés  ici 
		  
		  if (debugIn) PApplet.println("Val(0,0) ="+opencv_core.cvGet2D(Trans32F3C, 0,0).val(0)); // lit la valeur... utilise fonction .val(i) du scalar renvoyé par cvGet2D
		  // val(0) renvoie canal Bleu, val(1) le canal vert et val(2) le canal Rouge

		  if (debugIn) PApplet.println("Val(0,0) ="+opencv_core.cvGet2D(Trans16S3C, 0,0).val(2)); // lit la valeur... utilise fonction .val(i) du scalar renvoyé par cvGet2D

				  
		  opencv_core.cvMul(Trans16S3C, Trans32F3C, Trans16S3C, 1); // multiplie les 2 images 
		  // ce qui réalise pour chaque pixel : R x coeff R , G x coeff G, B x coeff B
		  //-- nb ; pour la multiplication avec des float, il faut que l'image soit une 32F

		  if (debugIn) PApplet.println("Val(0,0) ="+opencv_core.cvGet2D(Trans16S3C, 0,0).val(0)); // lit la valeur... utilise fonction .val(i) du scalar renvoyé par cvGet2D
		  

		  //-- dissociation des canaux de l'image source en 3 IplImages mono-canal
		  if (debugIn) PApplet.print("split : debut = "+p.millis()); // info
		  //opencv_core.cvSplit(iplImgSrc, iplImgR, null, null, null); 
		  //-- on utilise Trans16S1C pour R , Trans16S1C1 pour G et Trans16S1C pour B
		  opencv_core.cvSplit(Trans16S3C, Trans16S1C2, Trans16S1C1, Trans16S1C, null); // extrait les 3 canaux R G B : attention inversion RGB  
		  if (debugIn) PApplet.println(" | fin = "+p.millis()); // info

		  if (debugIn) PApplet.println("Val R (0,0) ="+opencv_core.cvGet2D(Trans16S1C, 0,0).val(0)); // lit la valeur... utilise fonction .val(i) du scalar renvoyé par cvGet2D
		  if (debugIn) PApplet.println("Val G (0,0) ="+opencv_core.cvGet2D(Trans16S1C1, 0,0).val(0)); // lit la valeur... utilise fonction .val(i) du scalar renvoyé par cvGet2D
		  if (debugIn) PApplet.println("Val B (0,0) ="+opencv_core.cvGet2D(Trans16S1C2, 0,0).val(0)); // lit la valeur... utilise fonction .val(i) du scalar renvoyé par cvGet2D

		  //-- calcul du canal de sortie rouge par sommation des 3 canaux avec coefficients 
		  //-- correspond à l'addition R = (R x coeff R) + (G x coeff G) + (B x coeff B)
		  opencv_core.cvAdd(Trans16S1C, Trans16S1C1, Trans16S1C, null); // additionne les 2 IplIMage dans un troisième - ici, réalise R = (R x coeff R) + (G x coeff G)
		  opencv_core.cvAdd(Trans16S1C, Trans16S1C2, Trans16S1C, null); // additionne les 2 IplIMage dans un troisième - ici, réalise R = (R x coeff R) + (G x coeff G)

		  if (debugIn) PApplet.println("Val B (0,0) ="+opencv_core.cvGet2D(Trans16S1C, 0,0).val(0)); // lit la valeur... utilise fonction .val(i) du scalar renvoyé par cvGet2D

		  //conversion d'un Objet Ipl16S en 8U mais sans changer la valeur
		  // opencv_core.cvConvertScale(iplImg16S,iplImg8U,(1.0/256.0),128);  conversion 16S vers 8U avec adaptation valeur pleine échelle 
		  opencv_core.cvConvertScale(Trans16S1C,Trans8U1C,1,0); // sans changer la valeur

		  if (debugIn) PApplet.println("Val B (0,0) ="+opencv_core.cvGet2D(Trans8U1C, 0,0).val(0)); // lit la valeur... utilise fonction .val(i) du scalar renvoyé par cvGet2D


		  //-- récupère les canaux G et B de départ --- 
		  opencv_core.cvSplit(iplImgIn, Trans8U1C2, Trans8U1C1, null, null); // extrait les 2 canaux G B : attention inversion RGB  
		  // on ne récupère pas ici le canal R car on l'a recalculé
		  

		  //-- reconstruction des canaux de l'image source à partir des 3 IplImages mono-canal  
		  if (grayOut) {
			  // sort les 3 canaux Idem 
			  opencv_core.cvMerge(Trans8U1C,Trans8U1C,Trans8U1C, null, iplImgIn); // reconstruction IplImage destination à partir des canaux départ.. - attention inversion RGB 
			  // Les canaux B et G sont inchangés et le canal R = = (R x coeff R) + (G x coeff G) + (B x coeff B)
			  
		  }
		  else { // si pas sortie en niveau de gris 
		  opencv_core.cvMerge(Trans8U1C2,Trans8U1C1,Trans8U1C, null, iplImgIn); // reconstruction IplImage destination à partir des canaux départ.. - attention inversion RGB 
		  // Les canaux B et G sont inchangés et le canal R = = (R x coeff R) + (G x coeff G) + (B x coeff B)
		  }
		  
		return(iplImgIn); 
	}
	
	//--- fonction mixerRGB() minimale : applique le mixer au Buffer avec des paramètres par défaut --- 
	public void mixerRGB() {
		 mixerRGB (Buffer, (float)1.0, (float)1.5, (float)-2.0, 0, false, false);
	}

	//--- fonction mixerRGBGray() avec coeff : applique le mixer au IplImage avec des paramètres par défaut et renvoie buffer en niveau gris --- 
	public void mixerRGB(opencv_core.IplImage iplImgIn, float coeffRIn, float coeffGIn, float coeffBIn) {
		 mixerRGB (iplImgIn, coeffRIn, coeffGIn, coeffBIn, 0, false, false);
	}

	//--- fonction mixerRGBGray() avec coeff : applique le mixer au Buffer avec des paramètres par défaut et renvoie buffer en niveau gris --- 
	public void mixerRGB(float coeffRIn, float coeffGIn, float coeffBIn) {
		 mixerRGB (Buffer, coeffRIn, coeffGIn, coeffBIn, 0, false, false);
	}

	//---- Fonction RGB Gray --- 

	//--- fonction mixerRGBGray() minimale : applique le mixer au Buffer avec des paramètres par défaut et renvoie buffer en niveau gris --- 
	public void mixerRGBGray() {
		 mixerRGB (Buffer, (float)1.0, (float)1.5, (float)-2.0, 0, true, false);
	}

	//--- fonction mixerRGBGray() avec coeff : applique le mixer au Buffer avec des paramètres par défaut et renvoie buffer en niveau gris --- 
	public void mixerRGBGray(float coeffRIn, float coeffGIn, float coeffBIn) {
		 mixerRGB (Buffer, coeffRIn, coeffGIn, coeffBIn, 0, true, false);
	}
	//--- fonction mixerRGBGray() avec coeff : applique le mixer au IplImage avec des paramètres par défaut et renvoie buffer en niveau gris --- 
	public void mixerRGBGray(opencv_core.IplImage iplImgIn, float coeffRIn, float coeffGIn, float coeffBIn) {
		 mixerRGB (iplImgIn, coeffRIn, coeffGIn, coeffBIn, 0, true, false);
	}

	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////// METHODES DE DETECTION DE CONTOURS /////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	

	/////////////////////////////////////   Détection de contour : Sobel natif  ///////////////////////////////////////////////////////////////

	//======= fonction sobel() : applique un filtre sobel (détection de contours) à une IplImage ===
	// --- applique séparément le filtre sobel horizontal puis vertical et combine les 2
	
	public opencv_core.IplImage sobel (opencv_core.IplImage iplImgIn, int ksizeIn, float scaleIn) { // la fonction reçoit le IplImage à traiter et  renvoie le IplImage modifié

		//--- ici, on calcule d'une part le Sobel Gx puis le Sobel Gy
		//--- le passage par les 2 canaux séparés donne un bien meilleur résultat que Sobel 1,1 pour x et y simultanés 
		
		//javacv : static void 	cvSobel(opencv_core.CvArr src, opencv_core.CvArr dst, int xorder, int yorder, int aperture_size) 
		// où :
		// src et dst sont 2 images IplImage source et destination - nécessite une source en 8U et une destination en 16S... 
		// xorder : à 1 si Sobel horizontal
		// yorder : à 1 si Sboel vertical
		// aperture_size = taille du noyau utilisé - fixé par paramètre KsizeIn reçu par la fonction 
	
		// NB : La fonction cvSobel effectue un Sobel Normalisé çàd divise coeff  noyau / taille noyau
		// pour Sobel avec noyau non normalisé, voir SobelBrut() de cette librairie 
		
		// le paramètre scaleIn joue comme un coeff faisant varier l'intensité du pourtour - utilisé pour basculer de 16S vers 8U
	  		
		//--- Sobel horizontal -- 
		opencv_imgproc.cvSobel(iplImgIn, Trans16S3C, 1,0,ksizeIn); // applique une détection contour par filtre Sobel horizontal
		//--- attention le Sobel nécessite une source en 8U et une destination en 16S... 

		//--- Sobel vertical -- 
		opencv_imgproc.cvSobel(iplImgIn, Trans16S3C2, 0,1,ksizeIn); // applique une détection contour par filtre Sobel vertical
		//--- attention le Sobel nécessite une source en 8U et une destination en 16S... 

		//--- ensuite reconvertit l'image destination en 8 bits avec la fonction cvConvertScale
		//--- le sobel détecte des fronts en positif et en négatif : pour les prendre en compte, valeur absolue obligatoire

		opencv_core.cvConvertScaleAbs(Trans16S3C,Trans8U3C,scaleIn,0); // scale fixé par la fonction - pas de shift 
		opencv_core.cvConvertScaleAbs(Trans16S3C2,Trans8U3C2,scaleIn,0); // scale fixé par la fonction - pas de shift
		// NB : cvConvertScaleAbs() utilise obligatoirement une destination en 8U càd 8 bits non signés
		

		//---- addition des 2 images dans la même ----- 
		opencv_core.cvAdd(Trans8U3C, Trans8U3C2, iplImgIn, null); // additionne les 2 dans Sobel vertical et horizontal 
		 
		
		  return(iplImgIn); 
	  
	}
	
	//---- fonction sobel() par défaut : applique sobel au buffer IplImage
	public void sobel(){
		
		Buffer=this.sobel(Buffer,3,1);  // utilise par défaut kernel 3 et scale 1
	}

	//---- fonction sobel (ksize) : applique sobel au buffer IplImage
	public void sobel(int ksizeIn){
		
		Buffer=this.sobel(Buffer,ksizeIn,1);  // utilise par défaut  scale 1
	}

	//---- fonction sobel (ksize, scale) : applique sobel au buffer IplImage
	public void sobel(int ksizeIn, float scaleIn){
		
		Buffer=this.sobel(Buffer,ksizeIn, scaleIn);
	}
	
	// todo : ajouter sobel( stringIplImage, ksize, scale) ? 
	


	/////////////////////////////////////// Détection de contours : filtre de SCHARR - dérive de cvSobel() /////////////////////////////////////
	
	//------------- scharr() ---------------------
	//-- le filtre de SCHARR est une variante du filtre de Sobel avec un noyau un peu différent qui accentue davantage le contour
	// disponible directement par la fonction cvSobel en utilisant le ksize = CV_SCHARR (= -1)
	
	//--- scharr minimale sur buffer
	public void scharr() {
		
		sobel(Buffer, opencv_imgproc.CV_SCHARR, 1); // sobel avec ksize=opencv_imgproc.CV_SCHARR
	}

	//--- scharr sur buffer avec coeff
	public void scharr(float scaleIn) {
		
		sobel(Buffer, opencv_imgproc.CV_SCHARR,scaleIn); // sobel avec ksize=opencv_imgproc.CV_SCHARR
	}


	//--- scharr sur IplImage avec coeff
	public void scharr(opencv_core.IplImage iplImgIn, float scaleIn) {
		
		sobel(iplImgIn, opencv_imgproc.CV_SCHARR,scaleIn); // sobel avec ksize=opencv_imgproc.CV_SCHARR
	}


	///////////////////////////////////// Détection contour : Sobel V2  : fonction Sobel via cvFilter2D pour amélioration performances /////////////////////////////////////////////
	
	// fonction principale - Sobel 2 -
	public opencv_core.IplImage sobel2 (opencv_core.IplImage iplImgIn, int ksizeIn, float scaleIn, float coeffNormIn) { // la fonction reçoit le IplImage à traiter et  renvoie le IplImage modifié

		//--- ici, on calcule d'une part le Sobel Gx puis le Sobel Gy
		//--- le passage par les 2 canaux séparés donne un bien meilleur résultat que Sobel 1,1 pour x et y simultanés 
		

		  //--- initialisation des objets utiles pour le traitement d'image par noyau de convolution 
		  
		  int kernelSize=3; // pour kernel 3x3
		  
		  FloatPointer myPtr = new FloatPointer(kernelSize*kernelSize); // crée un pointeur de (kernelSize x kernelSize) floats pour le noyau 
		   
		  float value=(float)0.0; // variable calcul kernel normalisé - en fait pas vraiment utilisée ici 
		  // pour chaque élément du kerne (i,j), on aura : value = (1/kernelSize x kernelSize) * kernel[i][j] 
		   
		   //float coeffNorm=(float)4.0; // coeff pour accentuer le pourtour (= effectuer normalisation partielle du noyau..)
		  // coeffNormIn est reçu en paramètre 
		  
		   //--- création d'une matrice 2D pour le noyau, de taille kernelSize x kernelSize et de type Float 
		   //static opencv_core.CvMat 	cvMat(int rows, int cols, int type, com.googlecode.javacpp.Pointer data) 
		   opencv_core.CvMat matrix2D=  opencv_core.cvMat(kernelSize, kernelSize, opencv_core.CV_32F, myPtr) ; // crée un CvMat avec même taille de donnée unitaire
		   
		   //*************** chargement de l'image dans un buffer 16S ******************************
		   opencv_core.cvConvertScale(iplImgIn, Trans16S3C, 256.0, -32768); // convertit 8U en 16S

		   
		  //******************* application du noyau Sobel x **********************

		   //--- définition du noyau de convolution à utiliser ---

		  // --- kernel 3x3 Sobel x --  
		   float[][] kernelGx = {{+1,0,-1}, // déclaration des valeurs entières à utiliser
		                       {+2,0,-2},
		                       {+1,0,-1}};


		   //---- remplissage du kernel Gx----
		   for (int i=0; i < kernelSize; i++) { 
		     for (int j=0; j< kernelSize; j++) {
		       
		      value=coeffNormIn * kernelGx[i][j] / (kernelSize*kernelSize); // calcul valeur normalisée du kernel - coeffNorm atténue la normalisation et accentue le contour
		      //value=kernelGx[i][j]; // si utilisation de la valeur non normalisée. Peut donner meilleur résultat dans certains cas... cf Sobel
		      
		    //static void 	cvSet2D(opencv_core.CvArr arr, int idx0, int idx1, opencv_core.CvScalar value) 
		    opencv_core.cvSet2D(matrix2D, i, j, opencv_core.cvScalarAll(value)); // remplit le CvMat à l'index (i,j) voulu  avec la valeur normalisée
		    
		    // debug
		    //print ("Kernel ["+i+"] ["+j+"] = "+ kernelGx[i][j]); 
		    //println(" | Valeur normalisée ("+i+","+j+") ="+opencv_core.cvGet2D(matrix2D,i,j).val(0)); // lit la valeur (i,j)... utilise fonction .val(index) du scalar renvoyé par cvGet2D

		     } // fin j
		     
		   } //fin i
		   

		  
		  //--- application du noyau normalisé à l'image source --- 
		  //static void cvFilter2D(opencv_core.CvArr src, opencv_core.CvArr dst, opencv_core.CvMat kernel, opencv_core.CvPoint anchor) 
		  opencv_imgproc.cvFilter2D(Trans16S3C, Trans16S3C1, matrix2D, opencv_core.cvPoint (-1,-1)) ; 
		   

		  //******************* application du noyau Sobel y **********************
		  
		  // --- kernel 3x3 Sobel y --  
		   float[][] kernelGy = {{+1,+2,+1}, // déclaration des valeurs entières à utiliser
		                       {0,0,0},
		                       {-1,-2,-1}};


		 

		   //---- remplissage du kernel Gy----
		   for (int i=0; i < kernelSize; i++) { 
		     for (int j=0; j< kernelSize; j++) {
		       
		      value=coeffNormIn * kernelGy[i][j] / (kernelSize*kernelSize); // calcul valeur normalisée du kernel - coeffNorm atténue la normalisation et accentue le contour
		      //value=kernelGy[i][j]; // si utilisation de la valeur non normalisée. Peut donner meilleur résultat dans certains cas... cf Sobel
		      
		    //static void 	cvSet2D(opencv_core.CvArr arr, int idx0, int idx1, opencv_core.CvScalar value) 
		    opencv_core.cvSet2D(matrix2D, i, j, opencv_core.cvScalarAll(value)); // remplit le CvMat à l'index (i,j) voulu  avec la valeur normalisée
		    
		    // debug
		    //print ("Kernel ["+i+"] ["+j+"] = "+ kernelGy[i][j]); 
		    //println(" | Valeur normalisée ("+i+","+j+") ="+opencv_core.cvGet2D(matrix2D,i,j).val(0)); // lit la valeur (i,j)... utilise fonction .val(index) du scalar renvoyé par cvGet2D

		     } // fin j
		     
		   } //fin i
		  
		  //--- application du noyau normalisé à l'image source --- 
		  //static void cvFilter2D(opencv_core.CvArr src, opencv_core.CvArr dst, opencv_core.CvMat kernel, opencv_core.CvPoint anchor) 
		  opencv_imgproc.cvFilter2D(Trans16S3C, Trans16S3C2, matrix2D, opencv_core.cvPoint (-1,-1)) ; 
		  

		  //******************* application des Sobel x et y dans une même image **********************  
		  
		  //---- combinaison des 2 gradients sobel vertical et horizontal dans la même image 
		  //static void cvAdd(opencv_core.CvArr src1, opencv_core.CvArr src2, opencv_core.CvArr dst, opencv_core.CvArr mask)  
		  // théoriquement, il faut faire sqrt(Gx² + Gy²)
		  // en pratique on peut approximer à |Gx|+|Gy|
		  
		  // NB : filter2D renvoie des images 8U non signés si on utilise des image 8U pour le filtre
		  // donc on "perd" les valeurs négatives fournies par le Sobel : appliquer par conséquent le Sobel sur valeur 16S
		  
		  opencv_core.cvConvertScaleAbs(Trans16S3C1, Trans8U3C,(scaleIn*1.0/256),0); // passer en valeur absolue et en 8 bits
		  opencv_core.cvConvertScaleAbs(Trans16S3C2, Trans8U3C2,(scaleIn*1.0/256),0); 

		  //opencv_core.cvConvertScale(iplImgTransGx, iplImg8UC3_Gx,1.0/256,64); // didactique - pour visualiser les fronts haut et bas renvoyés par Sobel 
		  //opencv_core.cvConvertScale(iplImgTransGy, iplImg8UC3_Gy,1.0/256,64); 
		  // le shift (dernière valeur de la fonction) fixe le niveau moyen de l'image - utiliser < 128 pour éviter image trop blanche... 
		  
		  //opencv_core.cvConvertScale(iplImgTransGx, iplImgTransGx,1.0/256,128); // non - passer par la valeur absolue et image 8U 
		  //opencv_core.cvConvertScale(iplImgTransGy, iplImgTransGx,1.0/256,128); 
		  
		  //opencv_core.cvAdd(iplImgTransGx, iplImgTransGy, iplImgDest,null); 
		  opencv_core.cvAdd(Trans8U3C, Trans8U3C2, iplImgIn,null); 
		 
		  //--- release CvMat -- 
		  //opencv_core.cvReleaseMat(matrix2D.ptr()); // libère mémoire utilisée par CvMat
		  matrix2D.release(); // alternative - ok 
		  
		  //--- renvoie l'image attendue --- 
		  
		  return(iplImgIn); 
	  
	}
	
	//---- fonction sobel2() - forme par défaut : applique sobel au buffer IplImage
	public void sobel2(){
		
		Buffer=this.sobel2(Buffer,3,1,1); // sur Buffer avec noyau 3x3, scaleIn x1  et coeffNorm x1
	}

	//---- fonction sobel2() - forme par défaut : applique sobel au buffer IplImage
	public void sobel2(float coeffNormIn){
		
		Buffer=this.sobel2(Buffer,3,1,coeffNormIn); // sur Buffer avec noyau 3x3, scaleIn x1  et coeffNorm x1
	}
	
	public void sobel2 (int ksizeIn, float scaleIn, float coeffNormIn) { // la fonction reçoit le IplImage à traiter et  renvoie le IplImage modifié

		Buffer=this.sobel2(Buffer,ksizeIn,scaleIn,coeffNormIn); // sur Buffer avec noyau ksizexksize, scaleIn  et coeffNorm  

	}
	
	/////////////////////////////////////// Détection de contours : algorithme de Canny ////////////////////////////////////////////////
	
	//======= fonction canny() : détection des contours par algorithme de Canny ==========

	public opencv_core.IplImage canny (opencv_core.IplImage iplImgIn, double threshold1In, double threshold2In, int ksizeIn) { // la fonction reçoit le IplImage à traiter et  renvoie le IplImage modifié

		if (iplImgIn.nChannels()==3) { // si reçoit une image à 3 canaux 
			opencv_imgproc.cvCvtColor(iplImgIn, BufferGray, opencv_imgproc.CV_RGB2GRAY); // bascule en niveaux de gris 
		}
		else { 
			opencv_core.cvCopy(iplImgIn, BufferGray); // sinon copie l'image dans le Buffer Gray
		}
		
		//---- applique algorithme Canny au BufferGray (1 canal - 8 bits)
		opencv_imgproc.cvCanny(BufferGray, BufferGray, threshold1In, threshold2In, ksizeIn);
		
		if (iplImgIn.nChannels()==3) { // si reçoit une image à 3 canaux 
				opencv_imgproc.cvCvtColor(BufferGray, iplImgIn, opencv_imgproc.CV_GRAY2RGB); // rebascule en RGB 
				// les 3 canaux du buffer RGB sont identiques = l'image est en niveaux de gris 
		}
		else { 
			opencv_core.cvCopy(BufferGray,iplImgIn); // sinon copie l'image dans le Buffer Gray			
		}

		  return(iplImgIn); 
	}
	
	
	//---- fonction canny : applique algorithme canny au IplImage avec noyau par défaut 
	public opencv_core.IplImage canny (opencv_core.IplImage iplImgIn, double threshold1In, double threshold2In) {
		
		return(canny(iplImgIn, threshold1In, threshold2In,3));
	}

	
	//---- fonction canny : applique algorithme canny au buffer IplImage
	public void canny(){
		
		Buffer=this.canny(Buffer, 100, 200,3);
	}

	//---- fonction canny : applique algorithme canny au buffer IplImage
	public void canny(double threshold1In, double threshold2In){
		
		Buffer=this.canny(Buffer, threshold1In, threshold2In,3);
	}
	
	//---- fonction canny : applique algorithme canny au buffer IplImage
	public void canny(double threshold1In, double threshold2In, int ksizeIn){
		
		Buffer=this.canny(Buffer, threshold1In, threshold2In,ksizeIn);
	}

	
	/////////////////////////////////// OPERATIONS ENTRE DEUX IMAGES /////////////////////////////////
	
	//--- fonction absDiff : réalise la différence absolue entre 2 images ----
	
	public void absDiff() {
		
		//opencv_core.cvCopy(iplImgIn, BufferGray); // copie l'image Ipl en entrée dans le Buffer Gray
		
		opencv_core.cvAbsDiff(Memory, Buffer, Memory2); // Réalise la différence absolue des images entre Memory et Buffer 
		// et stocke dans mémory 2
		// exemple type = soustraction contours
		
	} // fin absDiff
	
	
	



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   METHODES DE GESTION DES BLOBS (contours détectés) //////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	public Blob[] blobs(opencv_core.IplImage iplImgIn, long minAreaIn, long maxAreaIn, int maxBlobIn, boolean findHolesIn, int maxVerticesIn, boolean debug) { // la fonction Blobs renvoie un tableau de blobs
		
		//fonction lib ubaa : blobs( minArea, maxArea, maxBlobs, findHoles, MAX_VERTICES )
		
		// La fonction reçoit les paramètres suivants : 
		// minArea : l'aire minimum d'un Blob 
		// marArea : l'aire maximum d'un Blob
		// maxBlob : le nombre maximum de Blob à prendre en compte
		// findHoles : détermine si prise en compte ou pas de l'intérieur - true = recherche contours intérieurs et extérieurs
		// maxVertices : le nombre maximum de sommets = de point, d'un Blob unitaire - utiliser dans cvApproxPoly ? 
		// debug : pour afficher détails des points détectés
		
		// maxBlobIn : nombre maximum de Blob que la fonction doit renvoyer
		
		// la fonction blobs() va rechercher les contours de l'image bufferGray grâce à la fonction javacv cvFindContours()
		// les contours vont être renvoyés sous forme d'un tableau d'objet Blob
		// un objet Blob comportera toutes les informations d'un contour unitaire, notamment : 
		// - le tableau des points du contour
		// - le rectangle encadrant, le point central, l'aire, 
		// 
		// le nombre de Blob total sera accessible par nomTableau.length
		
		//--- variables utilisées par la méthode --- 
		
		//--- les éléments du blob courant ---
		int indiceContourBlob=0; // indice du Blob courant dans la séquence des contours CvSeq
	    float areaBlob=0; // aire du blob
	    float lengthArcBlob=0; // longueur du blob
	    float lengthBlob=0; // nombre de points du blob
	    
	    Point centroidBlob		= new Point(); // le point du centre du Blob

	    Point[] pointsBlob; // le tableau des points du Blob
	    
	    Rectangle rectangleBlob	= new Rectangle(); // le rectangle entourant le Blob

		//-- comptage des blobs --
		//int MAX_NUMOF_BLOBS=1000; // nombre max de blob à détecter 
		
		int nbContours=0; // variable nombre de contours
		
		int mode; // mode utilisé pour la recherche de contour
		if (findHolesIn==false) mode=opencv_imgproc.CV_RETR_EXTERNAL; else mode=opencv_imgproc.CV_RETR_LIST; 
		
		int areaRef=0; // aire référence prise en compte des Blobs
		
		// -- selon le paramètre findHoles, on prend ou non en compte que les contours externes
		// voir : documentation OpenCV de la fonction cvFindContours du module imgproc
		
		  //int comptBlob=0; // variable de comptage des Blob


		//Blob[] myBlobs= new Blob[maxBlobIn]; // crée un tableau de taille fixe d'objets Blob 
		//ArrayList<Blob> myBlobsList = new ArrayList<Blob>(); // alternative à un tableau fixe = créer un ArrayList de Blob  (une sorte de tableau à taille variable)
		myBlobsList = new ArrayList<Blob>(); // alternative à un tableau fixe = créer un ArrayList de Blob  - déclaré ici en static - cf entête classe 

		//--- objets utilisés par la méthode --- 
		//-- revoir utilisation static .. 

		//opencv_core.CvMemStorage storage= opencv_core.cvCreateMemStorage(0); // initialise objet conteneur CvMemStorage 
		//if (storage==null) storage= opencv_core.cvCreateMemStorage(0); // initialise objet conteneur CvMemStorage - initialise memstorage static - 1er appel blobs seulement 
		
		//static opencv_core.CvSeq 	cvCreateSeq(int seq_flags, int header_size, int elem_size, opencv_core.CvMemStorage storage)  
		//opencv_core.CvSeq contour= opencv_core.cvCreateSeq(int seq_flags, int header_size, int elem_size, opencv_core.CvMemStorage storage); // initialise objet CvSeq
		//opencv_core.CvSeq contours = new opencv_core.CvSeq(null); // initialise objet CvSeq vide - initialiser en tant qu'objet "null" - cvFindContours renverra un cvSeq rempli
		contours = new opencv_core.CvSeq(null); // initialise objet CvSeq static en tant qu'objet "null" - cvFindContours renverra un cvSeq rempli
		//opencv_core.cvClearSeq(contours); // vide le CvSeq...
		// contours va être utilisé pour récupérer le résultat de la fonction cvFindContours = séquence des séquences de contours
		  
		//--- cf déclaré en static
		//opencv_core.CvSeq[] contourCourant= new opencv_core.CvSeq[MAX_NUMOF_BLOBS]; // initialise un tableau d'objet CvSeq si n'existe pas 
		// contourCourant[i] va être utilisé pour stocker les contours unitaires à partir de la séquence de séquence CvSeq

		//if (contourCourant==null) { // si le tableau de CvSeq static est vide 
			//contourCourant= new opencv_core.CvSeq[MAX_NUMOF_BLOBS]; // initialise le tableau static d'objet CvSeq - seulement premier appel de blobs()
		//}// fin if
		
		//else { // sinon, on le vide - pas vraiment nécessaire... 
			
			// opencv_core.cvClearSeq(contourCourant); // vide le Seq... mais pas le tableau 
			
		//} // fin else - 
		

		//opencv_core.CvSeqBlock mySeqBlock; // crée un objet SeqBlock

		  opencv_core.CvPoint myPoint = new opencv_core.CvPoint(); // déclare un oblet CvPoint
		  
		  opencv_core.CvSeqReader reader= new opencv_core.CvSeqReader(); // déclare un objet CvSeqReader qui sera utilisé pour la lecture de l'objet CvSeq

		//---- chargement préalable de l'image dans le buffergray -----
		gray(iplImgIn); 
		  
		//--- détection des contours -à partir de l'image dans le Buffer Gray --   

		  //-- appel de la fonction cvFindContours 
		  opencv_imgproc.cvFindContours(BufferGray, storage, contours, Loader.sizeof(opencv_core.CvContour.class),mode, opencv_imgproc.CV_CHAIN_APPROX_SIMPLE);
		  //opencv_imgproc.cvFindContours(BufferGray, storage, contours, Loader.sizeof(opencv_core.CvContour.class),opencv_imgproc.CV_RETR_LIST, opencv_imgproc.CV_CHAIN_APPROX_SIMPLE);
		  //opencv_imgproc.cvFindContours(BufferGray, storage, contours, Loader.sizeof(opencv_core.CvContour.class),opencv_imgproc.CV_RETR_LIST, opencv_imgproc.CV_CHAIN_APPROX_NONE);
		  //opencv_imgproc.cvFindContours(BufferGray, storage, contours, Loader.sizeof(opencv_core.CvContour.class),opencv_imgproc.CV_RETR_EXTERNAL, opencv_imgproc.CV_CHAIN_APPROX_NONE);

		  // l'objet CvSeq contour contiendra une séquence de séquences de points (1 séquence pour le premier groupe de point, 1 séquence pour le 2ème, etc.. 
		  // cet objet CvSeq va contenir en fait une séquence de séquences de points, autrement dits, une séquence de contours... 
		  
		  // les séquences unitaires seront récupérées dans le tableau contourCourant[i]

		  
		  //---- passe en revue les contours retrouvés et extrait les infos nécessaires pour la création des blobs retenus
		  
		  if (debug) PApplet.println("****** Liste de tous les blobs détectés ******"); 
		  
		  //for (int k=0; k<maxBlobIn; k++) { // défile n contours
		  while (contours != null && !contours.isNull()&&(contours.elem_size() > 0)) { // tant qu'un contour existe et n'est pas de taille nulle
			  
			  // NB : le SeqBlock courant est désigné par le nom du CvSeq, ici contours. 
			  
			  nbContours++; // incrémente la variable de comptage du nombre de contours 
			  
			  contourCourant[nbContours-1]=contours; // récupère dans le CvSeq [i] du tableau CvSeq contourCourant la séquence courante de la séquence de séquences CvSeq contours
			  // ceci permettra ensuite un utilisation par indice des contours unitaires

			  //---- mémorise la variable contour
			  
			  
			  //--- aire du contour courant ---
			  indiceContourBlob=nbContours; // mémorise l'indice du contour dans le CvSeq
			  if (debug) PApplet.println("Indice contour Blob "+ nbContours+ " = " + indiceContourBlob); // affiche valeur aire contour
			  
			  //static double 	cvContourArea(opencv_core.CvArr contour, opencv_core.CvSlice slice, int oriented) 
			  double area=0; 
			  area=opencv_imgproc.cvContourArea(contourCourant[nbContours-1], opencv_core.CV_WHOLE_SEQ, 0 ); 
			  //if (debug) p.println("Aire contour "+ nbContours+ " = " + area); // affiche valeur aire contour

			  areaBlob=(float)area; // mémorise aire du blob

			  //--- longueur du contour courant --- 
			  double longueur=0; 
			  longueur=opencv_imgproc.cvArcLength(contourCourant[nbContours-1], opencv_core.CV_WHOLE_SEQ, -1 ); 
			  //if (debug) p.println("Longueur contour "+ nbContours+ " = " + longueur); // affiche valeur aire contour
			  
			  lengthArcBlob=(float)longueur; // longueur du blob

			  
			  //---- centre du contour courant et du blob ---- 
			  //-- pour calculer le centre d'un contour, on utilise ce que l'on appelle les moments du contour 
			  // ces moments sont des grandeurs s'apparentant au concept de barycentre d'un polygone 
			  // openCv fournit la fonction cvMoments qui calcule la valeur des moments jusqu'au 3ème ordre, noté M00, M01, M20, etc.. 
			  // le centre d'un contour, qui s'apparente au "centre de masse" (ou barycentre), se calcule à partir des moments et vaut |x| = M10/M00 et |y| = M01/M00
			  
			  opencv_imgproc.CvMoments moments=new opencv_imgproc.CvMoments(); // objet pour stocker les moments. 
			  
			  opencv_imgproc.cvMoments(contourCourant[nbContours-1], moments, 0); 
			  
			  //p.println ("Moments : M00="+moments.m00() +" | M01="+moments.m01() +" | M10="+moments.m10()); 
			  
			  centroidBlob.x=(int)(moments.m10()/moments.m00()); // calcule x du Point centroidBlob
			  centroidBlob.y=(int)(moments.m01()/moments.m00()); // calcule y du Point centroidBlob
			  
			  //if (debug) p.println ("Point centroidBlob : x="+centroidBlob.x +" | y="+centroidBlob.y ); 
			  
			  //---- rectangle autour du contour courant et du blob ----
			  opencv_core.CvRect myCvRect=opencv_imgproc.cvBoundingRect(contourCourant[nbContours-1], 0); // renvoie le plus petit rectangle entourant le contour
			  //P.println ("CvRect entourant : x="+myCvRect.x() +" | y="+myCvRect.y() +" || largeur="+myCvRect.width()+" | hauteur="+myCvRect.height()); 
			  
			  //p.noFill();
			  //p.stroke(0,255,255); //couleur contour dessin 
			  //p.rect(myCvRect.x(),myCvRect.y(),myCvRect.width(),myCvRect.height()); // dessine le rectangle
			  
			  //-- récupère le cvRect dans l'objet java Rectangle
			  rectangleBlob.x=myCvRect.x(); // récupère x du cvRect dans le champ x du Rectangle
			  rectangleBlob.y=myCvRect.y(); // récupère x du cvRect dans le champ x du Rectangle
			  rectangleBlob.height=myCvRect.height(); // récupère x du cvRect dans le champ x du Rectangle
			  rectangleBlob.width=myCvRect.width(); // récupère x du cvRect dans le champ x du Rectangle
			  
			  
			   //---Nombre de points du Blob (blob) ----- 
			   lengthBlob=contours.total(); // nombre de points du blob
			    
			  
			  //--- lecture des coordonnées des points de la séquence (du SeqBlock en fait) courante --- 
			  //static void 	cvStartReadSeq(opencv_core.CvSeq seq, opencv_core.CvSeqReader reader, int reverse) 
			  
			  opencv_core.cvStartReadSeq( contourCourant[nbContours-1], reader, 0 ); // initialisation lecture objet CvSeq 

			    
			   pointsBlob		= new Point[contours.total()]; // le tableau des points du Blob

			  for (int i=0; i<contours.total(); i++) { // défile les éléments du CvSeq courant càd du SeqBlock 

			      //static opencv_core.CvPoint 	CV_CURRENT_POINT(opencv_core.CvSeqReader reader) 
			      myPoint=opencv_core.CV_CURRENT_POINT(reader); // récupère les coordonnées du point à la position courante

			      //if (debug) p.print("Position reader : "+ opencv_core.cvGetSeqReaderPos(reader)); // position reader
			      
			      //if (debug) p.println(" | Point " + i + " : x= " + myPoint.x() + " : y = " + myPoint.y());  // récupère les coordonnées du point 
			      
			      
			      // ----- affichage du point dans la fenêtre Processing --- 
			      //p.stroke(255,255,0); // couleur jaune 
			      //p.point(myPoint.x(), myPoint.y()); // opencv utilise le meme schéma x,y que Processing 


			    //---- récupération des points du blob --- 
			    pointsBlob[i]= new Point(); // "C'est une erreur fréquente d'oublier de construire tous les éléments du tableau."
			    // sinon chaque élément du tableau pointe sur null - cf : http://www.ac-creteil.fr/util/programmation/java/cours_java/c-donnees2.html 

			    pointsBlob[i].x=myPoint.x(); // récupère la coordonnée de x du cvPoint dans le tableau points
			    pointsBlob[i].y=myPoint.y(); // récupère la coordonnée de y du cvPoint dans le tableau points
			    
			    //--- passer au point suivant du contour --- 
			    
			    //opencv_core.CV_NEXT_SEQ_ELEM(contour.elem_size(), reader); // avance d'un cran le reader 

			    //static void 	cvSetSeqReaderPos(opencv_core.CvSeqReader reader, int index, int is_relative) 
			    opencv_core.cvSetSeqReaderPos(reader, i+1, 0) ; // alternative : se positionne sur l'élément voulu suivant
			  
			  } // fin for i - fin défilement des points du contour
			  
			  
			  //if (debug)  p.println ("Blob "+ nbContours + " | aire :"+ areaBlob + " | long. :"+ lengthArcBlob + " | nb Points :"+ lengthBlob );// affiche résultat 
			  //if (debug)  p.println ("Blob "+  nbContours +" | centroid x :"+ centroidBlob.x+" | centroid y :"+ centroidBlob.y);

			  //---- ajoute un élément supplémentaire au tableau de Blob --- 
			  //Blob( float areaIn, float lengthArcIn, float lengthIn, Point centroidIn, Rectangle rectIn, Point[] pointsIn)
			  //myBlobs[nbContours-1]=new Blob(areaBlob, lengthArcBlob, lengthBlob, new Point (centroidBlob.x, centroidBlob.y), new Rectangle(rectangleBlob.width, rectangleBlob.height), pointsBlob); 
			  // attention : il faut passer un new Point [] pour bien passer la valeur et pas la référence... 
			  // idem pour rectangle
			  // ou sinon, faire un new à chaque passage dans la boucle... sinon, c'est le même... 
			  
			  // ajoute uniquement si aire dans les limites voulues 
			  //if ((areaBlob>=minAreaIn)&& (areaBlob<=maxAreaIn)){ 
			  areaRef=rectangleBlob.width *rectangleBlob.height; 			  
			  if (debug) PApplet.println("Blob "+ (nbContours-1) +" : areaRef="+areaRef);
			  //if (debug)p.println (" | Rect aire :"+  (rectangleBlob.width * rectangleBlob.height)); 


			  
			  if ((areaRef>=minAreaIn) && (areaRef<=maxAreaIn)){ // alternative plus cohérente : surface du rectangle encadrant 
			  
				  if (debug) PApplet.println("Blob"+ (nbContours-1) +" sélectionné"); 
				  
				  //alternative - ajoute un Blob à ArrayList de Blob
				  myBlobsList.add(new Blob(indiceContourBlob, areaBlob, lengthArcBlob, lengthBlob, new Point (centroidBlob.x, centroidBlob.y), new Rectangle(rectangleBlob.x, rectangleBlob.y, rectangleBlob.width, rectangleBlob.height), pointsBlob));
				  
				  if (myBlobsList.size()>=maxBlobIn) break; // sort de la boucle défilement contours unitaires
				  // si le nombre d'élément de la liste est supérieur ou égal au nombre de blob maxi autorisé
				
					
			  }
			  
			  //int j=nbContours-1; 

			  // debug array
			  //p.println ("Blob "+ j + " | aire :"+ myBlobs[j].area + " | long. :"+ myBlobs[j].lengthArc + " | nb Points :"+ myBlobs[j].length );// affiche résultat 
			  //p.println ("Blob "+  j +" | centroid x :"+ myBlobs[j].centroid.x+" | centroid y :"+ myBlobs[j].centroid.y);

			  // debug arayList 
			  //if (debug) p.println ("Blob "+ j + " | aire :"+ myBlobsList.area + " | long. :"+ myBlobs[j].lengthArc + " | nb Points :"+ myBlobs[j].length );// affiche résultat 
			  //if (debug) p.println ("Blob "+ j +" | centroid x :"+ myBlobs[j].centroid.x + " | centroid y :"+ myBlobs[j].centroid.y); 
		  
			  

			 contours = contours.h_next(); // passe à la séquence suivante (au BlockSeq suivant en fait ) 

			  
			  
		} // fin for k et/ou while - fin défilement des contours unitaires 
		
		  
		  //--- renvoi de la fonction ---
		  
		  //String[] sl = (String[]) list.toArray(new String[0]); // exemple de référence 
		  Blob[] myBlobs = (Blob[]) myBlobsList.toArray(new Blob[0]);// récupère le ArrayList dans le tableau de Blob 
		  
		  if (debug) PApplet.println("****** Liste et détail des blobs sélectionnés *****"); 
		  
		  //-- debug - vérification du contenu du tableau obtenu par toArray - 
		  for (int j=0; j<myBlobs.length; j++) { // debug - passe en revue les éléments du tableau 

			  if (debug){ 
			  PApplet.println ("Blob "+ j ); 
			  PApplet.println ( " | aire :"+ myBlobs[j].area + " | long. :"+ myBlobs[j].lengthArc + " | nb Points :"+ myBlobs[j].length );// affiche résultat 
			  PApplet.println (" | centroid x :"+ myBlobs[j].centroid.x + " | centroid y :"+ myBlobs[j].centroid.y); 
			  PApplet.print (" | Rect x :"+ myBlobs[j].rectangle.x +" | Rect y :"+ myBlobs[j].rectangle.y + " | Rect w :"+ myBlobs[j].rectangle.width + " | Rect h :"+ myBlobs[j].rectangle.height); 
			  PApplet.println (" | Rect aire :"+  (myBlobs[j].rectangle.width * myBlobs[j].rectangle.height)); 

			  PApplet.println (" | Nombre points :"+ myBlobs[j].points.length); 
			  
			  for (int l=0; l<myBlobs[j].points.length; l++) {
				  PApplet.print("("+myBlobs[j].points[l].x+","+myBlobs[j].points[l].y+") | ");
			  } // fin for l 
			  
			  PApplet.println();

		  
		  	} // fin for j 
			  
		  } // fin if debug
		  
		return (myBlobs); // renvoie le tableau de Blob
		// nb : le nombre de Blob du tableau est accessible par tableau.length
		
		
	} // fin de la méthode Blobs 
	
	//--- syntaxe idem lib OpenCV ubaa ---
	public Blob[] blobs(int minAreaIn, int maxAreaIn, int maxBlobIn, boolean findHolesIn, int maxVerticesIn) { // la fonction Blobs renvoie un tableau de blobs
		//fonction lib ubaa : blobs( minArea, maxArea, maxBlobs, findHoles, MAX_VERTICES )
		
		return ( blobs(minAreaIn, maxAreaIn, maxBlobIn, findHolesIn, maxVerticesIn, false)); // fonction blobs avec debug false
		
	}
	
	//---- fonction principale sans IplImage = Buffer par défaut --- 
	public Blob[] blobs(long minAreaIn, long maxAreaIn, int maxBlobIn, boolean findHolesIn, int maxVerticesIn, boolean debug) { // la fonction Blobs renvoie un tableau de blobs

		return(blobs(Buffer, minAreaIn, maxAreaIn, maxBlobIn, findHolesIn, maxVerticesIn, debug)); 
	}

	//--- idem principale avec int en paramètres d'aires
	public Blob[] blobs(int minAreaIn, int maxAreaIn, int maxBlobIn, boolean findHolesIn, int maxVerticesIn, boolean debug) { // la fonction Blobs renvoie un tableau de blobs

		return(blobs((long) minAreaIn, (long) maxAreaIn, maxBlobIn, findHolesIn, maxVerticesIn, debug)) ;// appelle fonction blobs principale

	}
	//--- blobs avec affichage de debug --- 
	public Blob[] blobs(boolean debugIn) { // la fonction Blobs renvoie un tableau de blobs
		//fonction lib ubaa : blobs( minArea, maxArea, maxBlobs, findHoles, MAX_VERTICES )
		
		// fonction blobs avec debug true et param pour détection tous les blobs
		return ( blobs(area()/256, area(), 1000, true, OpenCV.MAX_VERTICES*4, debugIn)); 
		
	}
	
	//--- fonction blobs() minimale - par défaut 
	public Blob[] blobs() { // la fonction Blobs renvoie un tableau de blobs

		return(blobs(area()/256, area(), 1000, true, OpenCV.MAX_VERTICES*4, false)) ;// appelle fonction blobs principale

	}
	
	///////////////////////////////// Fonction selectBlobs ///////////////////////////////////////////////////
	// ces fonctions réalise une sélection des blobs selon certains critères : ratio H/W du blob et ratio Area forme / area rect
	public Blob[] selectBlobs(Blob[] blobsIn, boolean hwTestIn, float ratioWHTest, float ratioHWTest, boolean areaTestIn, float areaRatioTest, boolean modeIn, boolean debugIn) {
		
		// La fonction reçoit un tableau de Blob, analyse 1 à 1 les Blob du tableau  et sélectionne certains blobs en fonction des critères fixés
		// la fonction peut tester le ratio h/w du rectangle entourant ce qui permet de ne garder que les blobs carrés ou proche du carré
		// la fonction peut tester également le ratio de l'aire du blob / l'aire du rectangle ce qui permet de ne garder que les formes proches du cercle par exemple
		
		// La fonction reçoit : 
		// blobsIn : un tableau de blobs, typiquement obtenu à l'aide de la fonction blobs()
		// hwTestIn : le drapeau d'activation du test h/w - Actif si true, inactif si false
		// ratioWHTest : valeur de référence pour le ratio w/h
		// ratioHWTest : valeur de référence pour le test h/w
		// Nb : le test des 2 ratio H/W et W/H est testé simultanément (OU)
		// areaTestIn : le drapeau d'activation du test aire Blob / aire Rect 
		// areaRatio : valeur de référence pour le ratio aire Blob / aire Rect
		// modeIn : drapeau fixant le test à réaliser - si les conditions sont vraies si true - si les conditions sont fausses si false 
		// debugIn : drapeau d'activation des messages de debug de la fonction 
		
		ArrayList<Blob> selectBlobsList = new ArrayList<Blob>(); // ArrayList des Blobs sélectionnés  (une sorte de tableau à taille variable)
		
		Rectangle rectangleBlob;  // objet rectangle utilisé par la fonction 
		
		for (int i=0; i<blobsIn.length; i++ ) { // passe en revue les Blob du tableau reçu par la fonction
			
	        rectangleBlob=blobsIn[i].rectangle; // récupère le rectangle qui contient la forme détectée
	        
	        //---- analyse du rectangle objet ---- 
	        int ratioWH=rectangleBlob.width/rectangleBlob.height; // calcule le ratio largeur/hauteur
	        int ratioHW=rectangleBlob.height/rectangleBlob.width; // calcule le ratio hauteur/largeur

	        float aireBlob=blobsIn[i].area; // récupère l'aire de la forme courante
	        
	        
	        float ratioAire=aireBlob/(rectangleBlob.width*rectangleBlob.height); // calcul du ratio Aire forme / Aire rectangle
	        // aire cercle = pi * r²
	        // aire carré = (2r)²=4r²
	        // aire cercle/aire carré = (pi * r²)/ (4*r²) = pi/4 = 3/4 env
	        // un cercle occupe 3/4 du carré le contenant
	        
	        //--- si test HW et test Aire simultanés 	        
	        if ((hwTestIn) && (areaTestIn)) { // si test HW et test area activés simmultanément 
	        
	        	//if ( ( (ratioWH>ratioWHTest) || (ratioHW>ratioHWTest) )  && (ratioAire>areaRatioTest)  ) { 
	        	if ( 
	        		((ratioWH<(1+(1-ratioWHTest)))&&(ratioWH>ratioWHTest)&&(ratioAire>areaRatioTest)) // 1 +/- x%
	        		//|| ((ratioHW<1)&&(ratioHW>ratioHWTest)&&(ratioAire>areaRatioTest))
	        			) { 
	        		// si ratio W/H conforme ET si ratio aireconforme

	        		//ajoute un Blob à ArrayList de Blob
		        	selectBlobsList.add(blobsIn[i]); // ajoute le blb à l'ArrayList 

	        	} // si ratio W/H conforme ET si ratio W/H conforme
		        
	        } // fin si test HW et test area activés simmultanément 

	        
	        //--- si test HW actif et test Aire inactif  	        
	        if ((hwTestIn) && (!areaTestIn)) { // si test HW actif et test Aire pas actif
	        
	        	if (  (ratioWH>ratioWHTest) || (ratioHW>ratioHWTest) ) { 
	        		// si ratio W/H conforme 

	        		//ajoute un Blob à ArrayList de Blob
		        	selectBlobsList.add(blobsIn[i]); // ajoute le blb à l'ArrayList 

	        	} // si ratio W/H conforme 
		        
	        } // fin si test HW actif et test area inactif
	        

	        //--- si test HW inactif et test Aire actif  	        
	        if ((!hwTestIn) && (areaTestIn)) { // si test HW inactif et test Aire actif
	        
	        	if ((ratioAire>areaRatioTest) ) { 
	        		// si ratio aire conforme 

	        		//ajoute un Blob à ArrayList de Blob
		        	selectBlobsList.add(blobsIn[i]); // ajoute le blb à l'ArrayList 

	        	} // si ratio W/H conforme 
		        
	        } // fin si test HW inactif et test area actif

	        /*
	         
	        if ( ( (hwTestIn) && ( (ratioWH>ratioWHTest) || (ratioHW>ratioHWTest) ) ) && ( (areaTestIn) && (ratioAire>areaRatioTest) ) ) { 
	        	// si test HW actif ET si ratio W/H conforme
	        	// OU si test aire actif ET si le rapport de l'aire de la forme / rectangle contenant est conforme

				//ajoute un Blob à ArrayList de Blob
				//myBlobsList.add(new Blob(indiceContourBlob, areaBlob, lengthArcBlob, lengthBlob, new Point (centroidBlob.x, centroidBlob.y), new Rectangle(rectangleBlob.x, rectangleBlob.y, rectangleBlob.width, rectangleBlob.height), pointsBlob));
	        	selectBlobsList.add(blobsIn[i]); // ajoute le blb à l'ArrayList 
	        	
	        	}// fin if --- 
			*/
	        
	        
		} // fin for i - défilement des Blob
		
		//--- récupère le ArrayList dans un tableau 
		//Blob[] myBlobs = (Blob[]) selectBlobsList.toArray(new Blob[0]);// récupère le ArrayList dans le tableau de Blob 
		Blob[] myBlobs = (Blob[]) selectBlobsList.toArray(new Blob[selectBlobsList.size()]);// récupère le ArrayList dans le tableau de Blob
		
		  //-- debug - vérification du contenu du tableau obtenu par toArray - 
		PApplet.println("Nombre initial Blobs : " + blobsIn.length); //-- affiche nombre initial de blobs 
		PApplet.println("Nombre Blobs sélectionnés : " + myBlobs.length); //-- affiche nombre de blobs sélectionnés
		
		// renvoie le tableau de Blobs
		return(myBlobs); 
		
	}
	
	//--- fonction dérivée selectBallBlobs() : sélectionne uniquement les balles 
	public Blob[] selectBallBlobs(Blob[]blobsIn) {
		
		//les Blobs ayant caractéristiques proche du carré et forme proche du cercle = sélection ball
		Blob[] myBlobsOut=selectBlobs(blobsIn, true, (float) 0.80, (float) 0.80, true, (float) 0.60, true, true); 
		
		 // renvoie le tableau des blobs sélectionnés
		 return(myBlobsOut); 
	}
	
	//////////////////////////////// Fonctions "drawCentroidBlobs" ////////////////////////////////
	// ces fonctions fournissent des outils de tracé pour l'ensemble des Blobs 
	
	////// --- drawCentroidBlobs --- 
	// fonction principale drawCentroidBlobs : trace le centre de tous les Blobs 
	public void drawCentroidBlobs (Blob[] blobsIn, int xRefIn, int yRefIn, float scaleIn, int radius, int colorStrokeIn, int strokeWeightIn, boolean fillIn, int colorFillIn) {
		
		// la fonction reçoit : 
		// le tableau de Blobs utile
		// les coordonnées de référence pour le tracé (coin sup gauche)
		// le facteur d'échelle éventuel à appliquer - mettre à 1 par défaut 
		// le rayon du cercle - indépendant du facteur d'échelle
		// la couleur du pourtour
		// si remplissage
		// couleur remplissage
		// épaisseur pourtour
		
	if (blobsIn!=null) { // si le tableau de Blob n'est pas vide 
		

	      for( int i=0; i<blobsIn.length; i++ ) { // passe en revue les blobs (= formes détectées)

	          //---- Récupération du centre du Blob courant ---- 
	          int centreX= blobsIn[i].centroid.x; // centroid renvoie un objet Point qui fournit x et y
	          int centreY= blobsIn[i].centroid.y; // centroid renvoie un objet Point qui fournit x et y    
	          
	          //---------- fixe les paramètres graphiques à utiliser -----------
	          if (fillIn) p.fill(colorFillIn) ; else p.noFill(); 
	          p.stroke(colorStrokeIn); // couleur verte
	          p.strokeWeight(strokeWeightIn); // épaisseur pourtour 

	          //---------- dessine un cercle autour du centre détecté -----------
	          p.ellipse (xRefIn+(centreX*scaleIn),yRefIn+(centreY*scaleIn), radius,radius);
	          

	      } // fin for 
	      
	} // fin if (blobsIn!=null)
	
	      
	} // fin drawCentroidBlobs
	
	
	//-- drawCentroidBlobs minimale -- 
	public void drawCentroidBlobs (Blob[] blobsIn) {

		//--- par défaut cercle bleu vide 
		drawCentroidBlobs ( blobsIn, 0, 0, 1, 10, p.color(0,0,255),3, false, 0) ;

	}
	
	//-- drawCentroidBlobs minimale + coordonnées et échelle -- 
	public void drawCentroidBlobs (Blob[] blobsIn,int xRefIn, int yRefIn, float scaleIn) {

		//--- par défaut cercle bleu vide 
		drawCentroidBlobs ( blobsIn, xRefIn, yRefIn, scaleIn, 10, p.color(0,0,255),3, false, 0) ;

	}

	
	//-- drawCentroidBlobs minimale + coordonnées et échelle et radius -- 
	public void drawCentroidBlobs (Blob[] blobsIn,int xRefIn, int yRefIn, float scaleIn, int radius) {

		//--- par défaut cercle vert vide 
		drawCentroidBlobs ( blobsIn, xRefIn, yRefIn, scaleIn, radius, p.color(0,0,255),2, false, 0) ;

	}

	////// --- drawRectBlobs --- 
	
	// fonction principale drawRectBlobs : trace le rectangle contenant de tous les Blobs 
	public void drawRectBlobs (Blob[] blobsIn, int xRefIn, int yRefIn, float scaleIn, int colorStrokeIn, int strokeWeightIn, boolean fillIn, int colorFillIn) {
		
		// la fonction reçoit : 
		// le tableau de Blobs utile
		// les coordonnées de référence pour le tracé (coin sup gauche)
		// le facteur d'échelle éventuel à appliquer - mettre à 1 par défaut 
		// la couleur du pourtour
		// si remplissage
		// couleur remplissage
		// épaisseur pourtour
		
		if (blobsIn!=null) { // si le tableau de Blob n'est pas vide 
			

		      for( int i=0; i<blobsIn.length; i++ ) { // passe en revue les blobs (= formes détectées)

		          //---- Récupération des paramètres du rectangle entourant l'objet courant ---- 
	              Rectangle rectangleBlob=blobsIn[i].rectangle; // récupère le rectangle qui contient la forme détectée
		          
		          
		          //---------- fixe les paramètres graphiques  -----------
		          if (fillIn) p.fill(colorFillIn) ; else p.noFill(); 
		          p.stroke(colorStrokeIn); // couleur verte
		          p.strokeWeight(strokeWeightIn); // épaisseur pourtour 
		          
		         //--- tracé du rectangle entourant le blob courant  
		         p.rect( xRefIn+(rectangleBlob.x*scaleIn), yRefIn+(rectangleBlob.y*scaleIn), rectangleBlob.width*scaleIn, rectangleBlob.height*scaleIn );

		      } // fin for 
		      
		} // fin if (blobsIn!=null)
	
	      
	} // fin drawRectBlobs principale 
		

	//---- fonction drawrectBlobs par défaut --- 
	public void drawRectBlobs (Blob[] blobsIn) {
		
		drawRectBlobs(blobsIn, 0,0,1,p.color(0,255,255),2,false,0);
		
	} // fin drawRectBlobs (Blob[] blobsIn)

	//---- fonction drawrectBlobs par défaut  avec coordonnées et échelle --- 
	public void drawRectBlobs (Blob[] blobsIn, int xRefIn, int yRefIn, float scaleIn) {
		
		drawRectBlobs(blobsIn, xRefIn,yRefIn,1,p.color(0,255,255),2,false,0);
		
	} // fin drawRectBlobs (Blob[] blobsIn, int xRefIn, int yRefIn, float scaleIn)

	////------------------ fonction drawBlobs : trace les points du pourtour de tous les Blobs ---------
	
	// fonction principale drawBlobs : trace les points du pourtour de tous les Blobs 
	public void drawBlobs (Blob[] blobsIn, int xRefIn, int yRefIn, float scaleIn, int radius, int colorStrokeIn, int strokeWeightIn, boolean fillIn, int colorFillIn, int mode) {
		
		// la fonction reçoit : 
		// le tableau de Blobs utile
		// les coordonnées de référence pour le tracé (coin sup gauche)
		// le facteur d'échelle éventuel à appliquer - mettre à 1 par défaut 
		// le rayon du cercle des points - indépendant du facteur d'échelle
		// la couleur du pourtour
		// si remplissage
		// couleur remplissage
		// épaisseur pourtour
		// mode de tracé : 0 : juste les  points sous forme de cercle/point (si radius=0), 1 : forme vertex
		
		// nb : pour pourtour complet sans remplissage faire mode=1 et fill = false

		if (blobsIn!=null) { // si le tableau de Blob n'est pas vide 
			

		      for( int i=0; i<blobsIn.length; i++ ) { // passe en revue les blobs (= formes détectées)

		          
		          //---------- fixe les paramètres graphiques à utiliser -----------
		          if (fillIn) p.fill(colorFillIn) ; else p.noFill(); 
		          p.stroke(colorStrokeIn); // couleur pourtour
		          p.strokeWeight(strokeWeightIn); // épaisseur pourtour 
		          
		         //--- tracé des points du  blob courant  
	
		          if (mode==1) {// tracé vertex
		        	  
		        	  // tracé des formes détectées
		          
		        	  p.beginShape(); // début tracé forme complexe
		          
		        	  for( int j=0; j<blobsIn[i].points.length; j++ ) { // parcourt tous les points du pourtour du blob
		        		  p.vertex( xRefIn+blobsIn[i].points[j].x, yRefIn+blobsIn[i].points[j].y); // tracé des points de la forme
		        	  } // fin for j 
		          		          
		        	  p.endShape(PConstants.CLOSE); // tracé forme complexe

		          } // fin if mode==1
		          
		          else if (mode==0){ // tracé des points 
		        	  
		        	  for( int j=0; j<blobsIn[i].points.length; j++ ) { // parcourt tous les points du pourtour du blob
		        	  
		        		  if (radius==0) { // si radius =0

		        			  //---------- dessine un point au point courant du Blob courant -----------
			        		  p.point (xRefIn+blobsIn[i].points[j].x,yRefIn+blobsIn[i].points[j].y);
		        		  
		        		  }
		        		  else { // sinon 
		        			  
		        		  //---------- dessine un cercle autour du point courant du Blob courant -----------
		        		  p.ellipse (xRefIn+blobsIn[i].points[j].x,yRefIn+blobsIn[i].points[j].y, radius,radius);

		        		  }
		        		  
		        		  
		        	  } // fin for j 
		          
		          }// fin else if mode==0
		          
		          
		      } // fin for i
		      
		} // fin if (blobsIn!=null)

	
	      
	} // fin drawRect principale 
		
	//--- drawBlobs minimale --- 
	public void drawBlobs (Blob[] blobsIn) {

		//--- trace par défaut avec pourtour rouge et remplissage jaune 
		drawBlobs(blobsIn,0,0, 1, 1,p.color(255,0,0), 2, true, p.color(255,255,0), 1);

	}

	//--- drawBlobs minimale + coordonnées --- 
	public void drawBlobs (Blob[] blobsIn, int xRefIn, int yRefIn, float scaleIn) {

		//--- trace par défaut avec pourtour rouge et remplissage jaune 
		drawBlobs(blobsIn,xRefIn,yRefIn, scaleIn, 1,p.color(255,0,0), 2, true, p.color(255,255,0), 1);

	}

	//////////////////////////////// Fonctions convexPoints ////////////////////////////
	
	public void drawConvexPoints(Blob[] blobsIn, int xRefIn, int yRefIn, float scaleIn, int radius, int colorStrokeIn, int strokeWeightIn, boolean fillIn, int colorFillIn, boolean debugIn ){
	// la fonction détecte les points de convexité d'un blob en se basant sur l'indice contour mémorisé de chaque Blob
	// la séquence des CvSeq[indice] a été mémorisée lors de la détection des Blob
		
		
		
		int indice; // variable indice contour courant  
		
		//--- objets utiles --- 
		opencv_core.CvSeqReader reader= new opencv_core.CvSeqReader(); // déclare un objet CvSeqReader qui sera utilisé pour la lecture de l'objet CvSeq
		opencv_core.CvPoint myPoint = new opencv_core.CvPoint(); // déclare un oblet CvPoint
		
		if (debugIn) PApplet.println("Nombre d'éléments du tableau de Blob ="+blobsIn.length); // affiche le nombre de blob détecté 

		for (int k=0; k<blobsIn.length; k++) { // passe en revue les Blob du tableau 
			
			indice=blobsIn[k].indiceContour;
			
			if (debugIn) PApplet.println("Le Blob "+ k +" a l'indice CvSeq = "+ indice); // affiche l'indice CvSeq du Blob
			
			
            //--- analyse de la convexité du contour courant en partant de la séquence CvSeq[indiceContour-1] --- 

            opencv_core.CvSeq convexHullSeqPoints = new opencv_core.CvSeq(null); // initialise objet CvSeq vide - initialiser en tant qu'objet "null" 

            // static opencv_core.CvSeq cvConvexHull2(opencv_core.CvArr input, com.googlecode.javacpp.Pointer hull_storage, int orientation, int return_points)
            convexHullSeqPoints=opencv_imgproc.cvConvexHull2(contourCourant[indice-1], null, opencv_imgproc.CV_CLOCKWISE, 1); //Sequence de points

            
            //--- lecture de la séquence de Points ---
            
            if (debugIn) PApplet.println("Nombre Points Hull =" + convexHullSeqPoints.total()); // affiche le nombre de points Hull
            if (debugIn) PApplet.println("Taille élément du CvSeq convexHull :"+convexHullSeqPoints.elem_size()); // taille élément base du CvSeq
                                    
           opencv_core.cvStartReadSeq( convexHullSeqPoints, reader, 0 ); // initialisation lecture objet CvSeq 

            //---- passe en revue les points répertoriés ----
            for (int i=0; i<convexHullSeqPoints.total(); i++) {

            	//static opencv_core.CvPoint 	CV_CURRENT_POINT(opencv_core.CvSeqReader reader) 
            	myPoint=opencv_core.CV_CURRENT_POINT(reader); // récupère les coordonnées du point à la position courante

                if (contourCourant[indice-1].total()>50) { // si plus de 50 points dans le contour courant 
              
                	if (debugIn) PApplet.print("point convexHull "+i +" ("+myPoint.x()+","+myPoint.y()+") | ");
                  
                	//----- fixe les paramètres graphiques --- 
                	if (fillIn) p.fill(colorFillIn);else p.noFill();
                   	
                	p.stroke(colorStrokeIn);
                	p.strokeWeight(strokeWeightIn);
                	
                    //-- dessine le point convexité détecté -- 
                	p.ellipse(xRefIn+(myPoint.x()*scaleIn), yRefIn+(myPoint.y()*scaleIn),radius,radius); 
                  
                } // fin if 

               //--- passer au point suivant du CvSeq --- 
    
    //opencv_core.CV_NEXT_SEQ_ELEM(contour.elem_size(), reader); // avance d'un cran le reader 

    //static void 	cvSetSeqReaderPos(opencv_core.CvSeqReader reader, int index, int is_relative) 
    opencv_core.cvSetSeqReaderPos(reader, i+1, 0) ; // alternative : se positionne sur l'élément voulu suivant
              
            } // fin for i 
            
            if (debugIn) PApplet.println(); 
            
		}
		

	} // fin fonction drawConvexPoint

	//-- drawConvexPoint minimale -- 
	public void drawConvexPoints(Blob[] blobsIn) {
		
		drawConvexPoints(blobsIn, 0, 0, 1, 5, p.color(0,0,255),1,false, 0, false );
	}
	
	//-- drawConvexPoint minimale + coordonnées -- 
	public void drawConvexPoints(Blob[] blobsIn, int xRefIn, int yRefIn, float scaleIn) {
		
		drawConvexPoints(blobsIn, xRefIn, yRefIn, scaleIn, 5, p.color(0,0,255),1,false, 0, false ); 
	}
	
	///// Fonction convexityDefects ///// 

	public ConvexityDefect[] convexityDefects(Blob blobIn, float minDepthIn, float angleMaxIn, boolean debugIn) {
		
		// la fonction détecte les creux d'un blob en recevant le Blob concerné
		// chaque creux (ou convexityDefect est caractérisé par :
		// un point de début 
		// un point de fin , 
		// un point de profondeur maximale 
		// et la valeur de la profondeur maximale
		
		// la fonction reçoit :
		// l'objet Blob à analyser (forme)
		// la valeur minimale de profondeur à prendre en compte
		// l'angle maximum à prendre en compte (en radians)
		// le drapeau de debug
			
			
			int indice; // variable indice contour courant  

			//--- crée un tableau pour stocker les Convexity Defect pris en compte
         	ArrayList<ConvexityDefect> myConvexityDefectsList = new ArrayList<ConvexityDefect>(); // alternative à un tableau fixe = créer un ArrayList de Blob  - déclaré ici en static - cf entête classe 
			
         	//--- les caractéristiques du convexity defect courant ---
         	Point startCD		= new Point(); // le point de début du ConvexityDefect
         	Point endCD		= new Point(); // le point de fin du ConvexityDefect
         	Point depthCD		= new Point(); // le point de profondeur maximale du ConvexityDefect
    	 
         	float valueCD		= 0; // la valeur de la profondeur maximale du ConvexityDefect

         	//--- valeur calculée
         	float distSE_CD=0; // distance entre points start et end
         	float distSD_CD=0; // distance entre points start et depth
         	float distDE_CD=0; // distance entre points depth et end

         	float angleSDE_CD=0; // angle SDE

         	
			//--- objets utiles --- 
			//opencv_core.CvSeqReader reader= new opencv_core.CvSeqReader(); // déclare un objet CvSeqReader qui sera utilisé pour la lecture de l'objet CvSeq
			//opencv_core.CvPoint myPoint = new opencv_core.CvPoint(); // déclare un oblet CvPoint
			
			if (debugIn) PApplet.println("Nombre de points du Blob ="+blobIn.points.length); // affiche le nombre de points du blob reçu 

			//for (int k=0; k<blobsIn.length; k++) { // passe en revue les Blob du tableau 
				
				indice=blobIn.indiceContour;
				
				if (debugIn) PApplet.println("Le Blob a l'indice CvSeq = "+ indice); // affiche l'indice CvSeq du Blob
				
				//=============== détection des points de convexité - par indices (nécessaire pour convexDefect() ===========
                opencv_core.CvSeq convexHullSeqIndice = new opencv_core.CvSeq(null); // initialise objet CvSeq vide - initialiser en tant qu'objet "null" 

                 // static opencv_core.CvSeq cvConvexHull2(opencv_core.CvArr input, com.googlecode.javacpp.Pointer hull_storage, int orientation, int return_points)
                 convexHullSeqIndice=opencv_imgproc.cvConvexHull2(contourCourant[indice-1], null, opencv_imgproc.CV_CLOCKWISE, 0); // Séquence des indices de points
                 // NB : indiceCourant est static

                 //--- Extraction des convexityDefects()
                 
                 opencv_core.CvSeq convexityDefectSeq = new opencv_core.CvSeq(null); // initialise objet CvSeq vide - initialiser en tant qu'objet "null" 

                 //cvConvexityDefects(opencv_core.CvArr contour, opencv_core.CvArr convexhull, opencv_core.CvMemStorage storage) 
                 convexityDefectSeq=opencv_imgproc.cvConvexityDefects(contourCourant[indice-1], convexHullSeqIndice, null); 
                 
                 //--- passage en revue des convexityDefect ---- 
            
                 	opencv_core.CvPoint myPointDepth = new opencv_core.CvPoint(); // déclare un oblet CvPoint
                 	opencv_core.CvPoint myPointStart = new opencv_core.CvPoint(); // déclare un oblet CvPoint
                 	opencv_core.CvPoint myPointEnd = new opencv_core.CvPoint(); // déclare un oblet CvPoint

 				if (debugIn) PApplet.println("Nombre de ConvexityDefect détecté dans le Blob "+ convexityDefectSeq.total()); // affiche le nombre de convexity defect détectés

                 
                 for (int i=0; i< convexityDefectSeq.total(); i++) { // passe en revue les convexity defect 

                   //-- récupère l'élément courant de la séquence --
                   // CvRect r = new CvRect(cvGetSeqElem(faces, i)); //-- exemple récupérer élément Seq à partir indice
                    opencv_imgproc.CvConvexityDefect myCD=new opencv_imgproc.CvConvexityDefect(opencv_core.cvGetSeqElem(convexityDefectSeq, i)); // récupère le convexity defect voulu de la séquence

            
                	//----- fixe les paramètres graphiques --- 
                	//if (fillIn) p.fill(colorFillIn);else p.noFill();
                   	
                	//p.stroke(colorStrokeIn);
                	//p.strokeWeight(strokeWeightIn);

                    
                	//--- la valeur de la profondeur maxi ---
                    
                    valueCD=myCD.depth();
                	if (debugIn) PApplet.print("ConvexityDefect "+ i +" : Profondeur = " + valueCD ); 
            

                    	//---- le point de profondeur maxi --- 

                    	myPointDepth=myCD.depth_point(); // récupère le point profondeur max
                    	
                    	depthCD.x=myPointDepth.x(); // récupère x du cvPoint dans Point
                    	depthCD.y=myPointDepth.y(); // récupère y du cvPoint dans Point
                    	
                    	if (debugIn)PApplet.print("Point Depth "+ i +" : x = " + depthCD.x +" : y = " + depthCD.y); 
            
                    	//---- le point End ----
                    	myPointStart=myCD.start(); // récupère le point de départ
 
                       	startCD.x=myPointStart.x(); // récupère x du cvPoint dans Point
                    	startCD.y=myPointStart.y(); // récupère y du cvPoint dans Point
 
                       	if (debugIn)PApplet.print("Point Start "+ i +" : x = " + startCD.x +" : y = " + startCD.y); 
                                   
                       	
                    	//--- le point Start --- 
                    	myPointEnd=myCD.end(); // récupère le point de départ
                       	
                    	endCD.x=myPointEnd.x(); // récupère x du cvPoint dans Point
                    	endCD.y=myPointEnd.y(); // récupère y du cvPoint dans Point
 
                       	if (debugIn)PApplet.print("Point End "+ i +" : x = " + endCD.x +" : y = " + endCD.y); 
            
                    	
                    //--- si le convexity Defect est pris en compte , on l'ajoute à la liste--- 
                    if (myCD.depth()>minDepthIn) { // si profondeur significative
                    	
                    	//--- on calcule les valeurs calculées utiles 
                    	
                    	//--- distance SE
                    	distSE_CD=distance(startCD,endCD); // calcule la distance entre les 2 points
                    	
                    	//--- distance SD
                    	distSD_CD=distance(startCD,depthCD); // calcule la distance entre les 2 points
 
                       	//--- distance DE
                    	distDE_CD=distance(depthCD,endCD); // calcule la distance entre les 2 points
                    	
                    	//--- angle SDE en radians --- 
                    	//public float calculAngleRadAlKashi( float adj1, float adj2, float opp) 
                    	angleSDE_CD=calculAngleRadAlKashi(distSD_CD,distDE_CD,distSE_CD);
                    	
                    	if (angleSDE_CD<=angleMaxIn) { // si on a un angle inf à angle maxi reçu par fonction
                    		
           				  //on ajoute un Blob à ArrayList de Blob
            				  myConvexityDefectsList.add(new ConvexityDefect( // -- attention : paramètres 1 à 1 par ligne pour clarté
            						  new Point (startCD.x, startCD.y), 
            						  new Point (endCD.x, endCD.y), 
            						  new Point (depthCD.x, depthCD.y), 
            						  valueCD, 
            						  distSE_CD, 
            						  distSD_CD, 
            						  distDE_CD, 
            						  angleSDE_CD) // fin new ConvexityDefect
            				  );// fin add List
            			
            				  if (debugIn) PApplet.print(" => sélectionné");
            				  
                    	} // fin if angleSDE
            				  
            	      	 
            	      		  
                    } // fin if profondeur significative
                    
                    
                    PApplet.println(); // entre 2 ConvexityDefect 
 
                 } // fin for défile convexity defect détectés
                 
                 
              //---- une fois défilement des CD fini : conversion de l'ArrayList en tableau
                 
       		  //String[] sl = (String[]) list.toArray(new String[0]); // exemple de référence 
       		  ConvexityDefect[] myConvexityDefects = (ConvexityDefect[]) myConvexityDefectsList.toArray(new ConvexityDefect[0]);// récupère le ArrayList dans le tableau de Blob 
       		  
       		  if (debugIn) PApplet.println("****** Liste et détail des Convexity Defect sélectionnés *****"); 
       		  
 
       		  //-- debug - vérification du contenu du tableau obtenu par toArray - 
       		  for (int j=0; j<myConvexityDefects.length; j++) { // debug - passe en revue les éléments du tableau 

       			  if (debugIn){ 
       			  PApplet.println ("Convexity Defect "+ j ); 
       			  PApplet.print (" : start x :"+ myConvexityDefects[j].start.x + " | start y :"+myConvexityDefects[j].start.y); 
       			  PApplet.print (" | end x :"+ myConvexityDefects[j].end.x + " | end y :"+myConvexityDefects[j].end.y); 
       			  PApplet.print (" | depth x :"+ myConvexityDefects[j].depth.x + " | depth y :"+myConvexityDefects[j].depth.y); 
       			  PApplet.print ( " | value :"+ myConvexityDefects[j].value);// affiche résultat 
      			  PApplet.println ( " | distSE :"+ myConvexityDefects[j].distSE + " | distSD :"+ myConvexityDefects[j].distSD + " | distDE :"+ myConvexityDefects[j].distDE);// affiche résultat 
       			  PApplet.println ( " | angleSDE:"+ myConvexityDefects[j].angleSDE +" radians soit " + PApplet.degrees(myConvexityDefects[j].angleSDE)+" degrés.");// affiche résultat 
      			       			  
       			  
       			  PApplet.println();
       			  
       			  } // fin if  debug
       			  
       			  } // fin for j
       			  
              //--- renvoi de la fonction =renvoie le tableau de Convexity Defect
       		  
                 return(myConvexityDefects); 
                 	            
	            
			//} // fin for k - défilement des Blob
	
	} // fin fonction convexityDefects
	
	
	///// Fonction detectFinger : analyse un tableau de ConvexityDefect issu d'une forme de main et renvoie le nombre de doigts détectés
	public int detectFinger(ConvexityDefect[] cdArrayIn, float angleMaxDoigtsIn, float angleMinPouceIn, float ratioPouceIndexIn, boolean debugIn) {
		
		//La fonction reçoit :
		//un tableau de Convexity Defect issu de l'analyse d'une forme de main par la fonction convexityDefects()
		// l'angle maximum (degrés) pour l'angle des doigts ouverts - mettre idem pouce min = 70° par défaut 
		// l'angle minimum (degrés) pour prise en compte pouce ouvert - 70° par défaut
		// NB : l'angle maxi pour le pouce ouvert est fixé par l'angle passé à la fonction convexityDefects()
		// le ratio à utiliser entre l'index ouvert et l'index fermé - 1.5 par défaut
		
		// la fonction renvoie un int qui correspond au nombre de doigts détectés : 
		// - renvoie 1 à 5 si pouce non séparable
		// - renvoie 10 à 13 si pouce séparable (par exemple 13 pour pouce + 3 doigts ouverts)
		// - renvoie 1 si 1 doigt sans pouvoir distinguer si pouce, 10 si pouce seul
		
		
		int nbFinger=0; 
		
        //--- analyse des convexity Defect retenus  pour interprétation doigts présents --- 
        
        //--- si 1 seul Convexity Defect retenu --- 
        if  (cdArrayIn.length==1) { // si un seul Convexity Defect retenu --- 
          
            if (cdArrayIn[0].angleSDE>PApplet.radians(angleMinPouceIn) ) { // si pouce ouvert 

                // et 1 autre doigt ouvert - analyse le ratio distSD / distDE - si doigt tendu, ce ratio vaut 2 environ
            	if ( (cdArrayIn[0].distDE/cdArrayIn[0].distSD > ratioPouceIndexIn) || (cdArrayIn[0].distSD/cdArrayIn[0].distDE > ratioPouceIndexIn))  {
            		if (debugIn) PApplet.println ("Pouce  + 1 autre doigt ouverts"); 
            		nbFinger=11;
            	}
            else {
            	if (debugIn) PApplet.println ("Pouce  seul ouvert");
            	nbFinger=10;
            }
            	  
           } // fin si pouce ouvert

            if(cdArrayIn[0].angleSDE<PApplet.radians(angleMaxDoigtsIn))  {
            	if (debugIn) PApplet.println ("Deux doigts ouverts");
            	nbFinger=2;
            }
            
            
        } // fin si un seul Convexity Defect Retenu

        //--- si 2 Convexity Defect retenu --- 
        if  (cdArrayIn.length==2) { // si 2 Convexity Defect retenu --- 
          
            if ( // si pouce + 2 doigts ouverts
                ( (cdArrayIn[0].angleSDE>PApplet.radians(angleMinPouceIn)) && (cdArrayIn[1].angleSDE<PApplet.radians(angleMaxDoigtsIn)) ) 
            ||  ( (cdArrayIn[1].angleSDE>PApplet.radians(angleMinPouceIn)) && (cdArrayIn[0].angleSDE<PApplet.radians(angleMaxDoigtsIn)) ) 

            ) {
            	if (debugIn) PApplet.println ("Pouce ouvert + 2 doigts ouverts"); 
            	nbFinger=12;
           }

            if ( // si 3 doigts ouverts hors pouce
                ( (cdArrayIn[0].angleSDE<PApplet.radians(angleMaxDoigtsIn)) && (cdArrayIn[1].angleSDE<PApplet.radians(angleMaxDoigtsIn)) ) 

            ) {
            	if (debugIn) PApplet.println ("3 doigts ouverts"); 
            	nbFinger=3;
           }
           
            
        } // fin si 2 Convexity Defect Retenu

        //--- si 3 Convexity Defect retenus --- 
        if  (cdArrayIn.length==3) { // si 3 Convexity Defect
          
            if ( // si pouce + 3 doigts ouverts
                ( (cdArrayIn[0].angleSDE>PApplet.radians(angleMinPouceIn)) && (cdArrayIn[1].angleSDE<PApplet.radians(angleMaxDoigtsIn)) && (cdArrayIn[2].angleSDE<PApplet.radians(angleMaxDoigtsIn))) 
            ||  ( (cdArrayIn[1].angleSDE>PApplet.radians(angleMinPouceIn)) && (cdArrayIn[2].angleSDE<PApplet.radians(angleMaxDoigtsIn)) && (cdArrayIn[0].angleSDE<PApplet.radians(angleMaxDoigtsIn)))
            ||  ( (cdArrayIn[2].angleSDE>PApplet.radians(angleMinPouceIn)) && (cdArrayIn[0].angleSDE<PApplet.radians(angleMaxDoigtsIn)) && (cdArrayIn[1].angleSDE<PApplet.radians(angleMaxDoigtsIn)))

            ) {
            	if (debugIn) PApplet.println ("Pouce ouvert + 3 doigts ouverts"); 
               	nbFinger=13;
                          }

            if ( // si 3 doigts ouverts hors pouce
                ( (cdArrayIn[0].angleSDE<PApplet.radians(angleMaxDoigtsIn)) 
                	&& (cdArrayIn[1].angleSDE<PApplet.radians(angleMaxDoigtsIn)) 
                	&& (cdArrayIn[2].angleSDE<PApplet.radians(angleMaxDoigtsIn)) ) 

            ) {
            	if (debugIn) PApplet.println ("4 doigts ouverts"); 
               	nbFinger=4;
                
            }
           
            
        } // fin si 3 Convexity Defect Retenu
        

        //--- si 4 Convexity Defect retenus --- 
        if  (cdArrayIn.length==4) { // si un seul Blob
          

        	if (debugIn) PApplet.println ("5 doigts ouverts"); // seule possibilité
          	nbFinger=5;
          	           
            
        } // fin si 4 Convexity Defect Retenu
        
		return(nbFinger);
	}
	
	
	// variante detectFinger simplifiée 
	
	public int detectFinger(ConvexityDefect[] cdArrayIn, boolean debugIn) {
		
		return (detectFinger(cdArrayIn, 70, 70, (float)1.5, debugIn));  // renvoie le int renvoyé par detectFinger
		
	}
		
	////// Fonction drawConvexityDefect
	public void drawConvexityDefect(Blob[] blobsIn, int xRefIn, int yRefIn, float scaleIn, int radius, int colorStrokeIn, int strokeWeightIn, boolean fillIn, int colorFillIn, int minDepthIn, boolean debugIn ){

		// la fonction détecte et dessinne les points de convexity defect d'un blob en se basant sur l'indice contour mémorisé de chaque Blob
			
			
			
			int indice; // variable indice contour courant  
			
			//--- objets utiles --- 
			//opencv_core.CvSeqReader reader= new opencv_core.CvSeqReader(); // déclare un objet CvSeqReader qui sera utilisé pour la lecture de l'objet CvSeq
			//opencv_core.CvPoint myPoint = new opencv_core.CvPoint(); // déclare un oblet CvPoint
			
			if (debugIn) PApplet.println("Nombre d'éléments du tableau de Blob ="+blobsIn.length); // affiche le nombre de blob détecté 

			for (int k=0; k<blobsIn.length; k++) { // passe en revue les Blob du tableau 
				
				indice=blobsIn[k].indiceContour;
				
				if (debugIn) PApplet.println("Le Blob "+ k +" a l'indice CvSeq = "+ indice); // affiche l'indice CvSeq du Blob
				
        //=============== détection des points de convexité - par indices (nécessaire pour convexDefect() ===========
                opencv_core.CvSeq convexHullSeqIndice = new opencv_core.CvSeq(null); // initialise objet CvSeq vide - initialiser en tant qu'objet "null" 

                 // static opencv_core.CvSeq cvConvexHull2(opencv_core.CvArr input, com.googlecode.javacpp.Pointer hull_storage, int orientation, int return_points)
                 convexHullSeqIndice=opencv_imgproc.cvConvexHull2(contourCourant[indice-1], null, opencv_imgproc.CV_CLOCKWISE, 0); // Séquence des indices de points
                 // NB : indiceCourant est static

                 //--- Extraction des convexityDefects()
                 
                 opencv_core.CvSeq convexityDefectSeq = new opencv_core.CvSeq(null); // initialise objet CvSeq vide - initialiser en tant qu'objet "null" 

                 //cvConvexityDefects(opencv_core.CvArr contour, opencv_core.CvArr convexhull, opencv_core.CvMemStorage storage) 
                 convexityDefectSeq=opencv_imgproc.cvConvexityDefects(contourCourant[indice-1], convexHullSeqIndice, null); 
                 
                 //--- passage en revue des convexityDefect ---- 
            
                 	opencv_core.CvPoint myPointDepth = new opencv_core.CvPoint(); // déclare un oblet CvPoint
                 	opencv_core.CvPoint myPointStart = new opencv_core.CvPoint(); // déclare un oblet CvPoint
                 	opencv_core.CvPoint myPointEnd = new opencv_core.CvPoint(); // déclare un oblet CvPoint

                 
                 for (int i=0; i< convexityDefectSeq.total(); i++) {

                   //-- récupère l'élément courant de la séquence --
                   // CvRect r = new CvRect(cvGetSeqElem(faces, i)); //-- exemple récupérer élément Seq à partir indice
                    opencv_imgproc.CvConvexityDefect myCD=new opencv_imgproc.CvConvexityDefect(opencv_core.cvGetSeqElem(convexityDefectSeq, i)); // récupère le convexity defect voulu de la séquence

            
                	//----- fixe les paramètres graphiques --- 
                	if (fillIn) p.fill(colorFillIn);else p.noFill();
                   	
                	p.stroke(colorStrokeIn);
                	p.strokeWeight(strokeWeightIn);

                	//---- traces les "convexity Defect" --------
                    
                	//--- la profondeur --- 
                	if (debugIn) PApplet.print("ConvexityDefect "+ i +" : Profondeur = " + myCD.depth() ); 
            
                	//---- le point de profondeur maxi --- 
                    if (myCD.depth()>minDepthIn) { // si profondeur significative
                    myPointDepth=myCD.depth_point(); // récupère le point profondeur max
                    if (debugIn)PApplet.print("Point Depth "+ i +" : x = " + myPointDepth.x() +" : y = " + myPointDepth.y()); 
            
                   
                    p.ellipse(xRefIn+(myPointDepth.x()*scaleIn), yRefIn+(myPointDepth.y()*scaleIn),radius,radius); 
                    
                    //---- le point End ----
                    myPointStart=myCD.start(); // récupère le point de départ
                    if (debugIn) PApplet.print("Point Start "+ i +" : x = " + myPointStart.x() +" : y = " + myPointStart.y()); 
            
                   
                    p.ellipse(xRefIn+(myPointStart.x()*scaleIn), yRefIn+(myPointStart.y()*scaleIn),radius,radius); 

                    //--- le point Start --- 
                    myPointEnd=myCD.end(); // récupère le point de départ
                    if (debugIn) PApplet.print("Point Start "+ i +" : x = " + myPointEnd.x() +" : y = " + myPointEnd.y()); 
            
                   
                    p.ellipse(xRefIn+(myPointEnd.x()*scaleIn), yRefIn+(myPointEnd.y()*scaleIn),5,5); 
                    
                    //--- trace ligne entre 2 points convexité
                    p.line (xRefIn+(myPointStart.x()*scaleIn), yRefIn+(myPointStart.y()*scaleIn), xRefIn+(myPointEnd.x()*scaleIn), yRefIn+(myPointEnd.y()*scaleIn)); 

                    } // fin if profondeur significative
                    
                    // continuer... 
                    
                     //convexityDefectSeq.h_next(); // passe à l'élément suivant 

                 } // fin for conexity defect
                 
                 PApplet.println(); 
	            
	            
			} // fin for k - défilement des Blob
			

		} // fin fonction drawConvexDefect

	//-- drawConvexityDefect : forme minimale -- 
	public void drawConvexityDefect(Blob[] blobsIn){
		
		drawConvexityDefect(blobsIn, 0, 0, 1, 5, p.color(255,0,255), 2, false,0, 20, true );

	}

	//-- drawConvexityDefect : forme minimale + coordonnée et échelle -- 
	public void drawConvexityDefect(Blob[] blobsIn, int xRefIn, int yRefIn, float scaleIn){
		
		drawConvexityDefect(blobsIn, xRefIn, yRefIn, scaleIn, 5, p.color(255,0,255), 2, false,0, 20, true );

	}

	//////// fonction drawConvexityDefects (avec un s) => dessine convexity à partir tableau de ConvexityDefects obtenu par convexityDefects() ////
	
	//---- forme principale ----- 
	public void drawConvexityDefects(ConvexityDefect[] cdArrayIn, int xRefIn, int yRefIn, float scaleIn, int radius, int colorStrokeIn, int strokeWeightIn, boolean fillIn, int colorFillIn,  boolean lineIn, int colorStrokeLineIn, int strokeWeightLineIn, int modeLineIn, boolean debugIn ){
		
		// la fonction reçoit :
		// un tableau de ConvexityDefect (obtenu avec la fonction convexityDefects()
		// le Blob source - non
		// les coordonnées de référence pour le tracé
		// l'échelle à utiliser
		// le rayon des points, la couleur et l'épaisseur du contour des points
		// le drapeau de remplissage des points et couleur remplissage
		// la couleur et épaisseur des lignes
		// le mode de traçage à utiliser - =0 que Start-End, =1 : 3 côtés
		// le drapeau de debug
		
		for (int n=0; n<cdArrayIn.length; n++) { // défile les Convexity Defect du tableau 
			
			//----- fixe les paramètres graphiques pour les points --- 
        	if (fillIn) p.fill(colorFillIn);else p.noFill();
           	
        	p.stroke(colorStrokeIn);
        	p.strokeWeight(strokeWeightIn);
        	
			//--- le point start ----
            if (debugIn) PApplet.println("Point Start "+ n +" : x = " + cdArrayIn[n].start.x +" : y = " + cdArrayIn[n].start.y); 
                        
            p.ellipse(xRefIn+(cdArrayIn[n].start.x*scaleIn), yRefIn+(cdArrayIn[n].start.y*scaleIn),radius,radius); 


			//--- le point end ----
            if (debugIn) PApplet.println("Point End "+ n +" : x = " + cdArrayIn[n].end.x +" : y = " + cdArrayIn[n].end.y); 
                        
            p.ellipse(xRefIn+(cdArrayIn[n].end.x*scaleIn), yRefIn+(cdArrayIn[n].end.y*scaleIn),radius,radius); 

			//--- le point depth ----
            if (debugIn) PApplet.println("Point Depth "+ n +" : x = " + cdArrayIn[n].depth.x +" : y = " + cdArrayIn[n].depth.y); 
                        
            p.ellipse(xRefIn+(cdArrayIn[n].depth.x*scaleIn), yRefIn+(cdArrayIn[n].depth.y*scaleIn),radius,radius); 

            
            if (lineIn) { // si dessin des lignes 

    			//----- fixe les paramètres graphiques pour les lignes --- 

            	p.stroke(colorStrokeLineIn);
            	p.strokeWeight(strokeWeightLineIn);
            	
            	// tracé ligne Start - End - dans tous les cas 
                p.line(xRefIn+(cdArrayIn[n].start.x*scaleIn), yRefIn+(cdArrayIn[n].start.y*scaleIn), xRefIn+(cdArrayIn[n].end.x*scaleIn), yRefIn+(cdArrayIn[n].end.y*scaleIn)); 

                if (modeLineIn==1) { // si mode = 1, on trace aussi les lignes Start - Depth et End - Depth 

                    p.line(xRefIn+(cdArrayIn[n].start.x*scaleIn), yRefIn+(cdArrayIn[n].start.y*scaleIn), xRefIn+(cdArrayIn[n].depth.x*scaleIn), yRefIn+(cdArrayIn[n].depth.y*scaleIn)); 
                    p.line(xRefIn+(cdArrayIn[n].end.x*scaleIn), yRefIn+(cdArrayIn[n].end.y*scaleIn), xRefIn+(cdArrayIn[n].depth.x*scaleIn), yRefIn+(cdArrayIn[n].depth.y*scaleIn)); 

                }


            	
            }
  
		} // fin for 
		
		
	} // fin drawConvexity  

	//--- forme minimale 
public void drawConvexityDefects(ConvexityDefect[] cdArrayIn) {
		
	drawConvexityDefects(cdArrayIn, 0,0,1,5, p.color(255,0,255), 2, false,0, true, p.color(255,0,0),2,0, true);
	
	}
	
	//-- forme allégée
	public void drawConvexityDefects(ConvexityDefect[] cdArrayIn, int xRefIn, int yRefIn, float scaleIn) {

		drawConvexityDefects(cdArrayIn, xRefIn,yRefIn,scaleIn,5, p.color(255,0,255), 2, false,0, true, p.color(255,0,0),2,0, true);

	}
	
	////////////////////////////////////// DETECTION D'OBJET - CASCADE CLASSIFIER ////////////////////////////////////////
	
	//----------------- Fonction cascade() : initialisation de la détection d'objet ---------------------------------
	// cette fonction charge le fichier de description de l'objet à rechercher 
	
	public void cascade(String classifierIn, boolean debug) {

		 //---- chemin / nom du  fichier .xml descriptif de l'objet recherché --- 

		// -- chemin absolu du fichier en fonction de l'OS 
		// nb : chemin variable selon plateforme
		
		String cheminAbsoluClassifier=""; 
		
		if ( OS.indexOf("mac")!=-1 ) {
			
			// à finir
			
		} // fin if mac

		else if ( OS.indexOf("windows")!=-1 ) {
			
			// à finir 
			
		} // fin if windows

		else if ( OS.indexOf("linux")!=-1 ) {
			cheminAbsoluClassifier="/usr/share/opencv/haarcascades/";
			if (debug) PApplet.println("Système Linux détecté ! Chemin par défaut utilisé : "+cheminAbsoluClassifier);			
		}
		  // ou "/usr/local/share/opencv/haarcascades/" (linux)
		

		//-- Nom du fichier à utiliser ---


		/*
		 * <li>CASCADE_FRONTALFACE_ALT_TREE</li>
		 * <li>CASCADE_FRONTALFACE_ALT2</li>
		 * <li>CASCADE_FRONTALFACE_DEFAULT</li>
		 * <li>CASCADE_PROFILEFACE</li>
		 * <li>CASCADE_FULLBODY</li>
		 * <li>CASCADE_LOWERBODY</li>
		 * <li>CASCADE_UPPERBODY</li>
		 */
		
		String fichierCascade=""; 
		
		if (classifierIn=="FRONTALFACE_ALT") fichierCascade="haarcascade_frontalface_alt.xml"; // cf le répertoire pour les différentes possibilités
		if (classifierIn=="FRONTALFACE_ALT_TREE") fichierCascade="haarcascade_frontalface_alt_tree.xml"; // cf le répertoire pour les différentes possibilités
		if (classifierIn=="FRONTALFACE_ALT2") fichierCascade="haarcascade_frontalface_alt2.xml"; // cf le répertoire pour les différentes possibilités
		if (classifierIn=="PROFILEFACE") fichierCascade="haarcascade_profileface.xml"; // cf le répertoire pour les différentes possibilités
		if (classifierIn=="FULLBODY") fichierCascade="haarcascade_fullbody.xml"; // cf le répertoire pour les différentes possibilités
		if (classifierIn=="LOWERBODY") fichierCascade="haarcascade_lowerbody.xml"; // cf le répertoire pour les différentes possibilités
		if (classifierIn=="UPPERBODY") fichierCascade="haarcascade_upperbody.xml"; // cf le répertoire pour les différentes possibilités
		//à compléter 
		
		//---- chargement du fichier voulu ----- 
		if (debug) PApplet.println ("Charge le fichier de description d'objet : " + cheminAbsoluClassifier + fichierCascade);
		
		cascade = new opencv_objdetect.CvHaarClassifierCascade(opencv_core.cvLoad( cheminAbsoluClassifier + fichierCascade ));

	} // fin fonction cascade 
	
	//---- cascade version chemin absolu fichier ---
	public void cascade(String cheminAbsoluRepIn, String nomFichierIn) {

		String lastCharRepIn=cheminAbsoluRepIn.substring(cheminAbsoluRepIn.length()-1,cheminAbsoluRepIn.length()); 
		PApplet.println (lastCharRepIn); // debug
		
		// --- gère l'absence du / en cas de chemin absolu 
		if ((PApplet.match(lastCharRepIn, "/")!=null)){
			//|| (PApplet.match(lastCharRepIn, Character.toString ((char) 92))!=null)) { // attention : le quadruple anti-slash correspon dau car anti-slash seul
			// Character.toString ((char) 92) pour le \ à partir du ascii
			
			// si le slash présent on utilise chaine telle quelle 
			PApplet.println ("Charge le fichier de description d'objet : " + cheminAbsoluRepIn + nomFichierIn);
			cascade = new opencv_objdetect.CvHaarClassifierCascade(opencv_core.cvLoad( cheminAbsoluRepIn + nomFichierIn));
		}
		else if ((OS.indexOf("linux")!=-1) || ( OS.indexOf("mac")!=-1 )){ // if linux ou Mac
			PApplet.println ("Charge le fichier de description d'objet : " + cheminAbsoluRepIn + "/"+ nomFichierIn);
			cascade = new opencv_objdetect.CvHaarClassifierCascade(opencv_core.cvLoad( cheminAbsoluRepIn + "/" + nomFichierIn));
		}
		else if ( OS.indexOf("windows")!=-1 ) {
			PApplet.println ("Charge le fichier de description d'objet : " + cheminAbsoluRepIn + "\\\\"+ nomFichierIn);
			cascade = new opencv_objdetect.CvHaarClassifierCascade(opencv_core.cvLoad( cheminAbsoluRepIn + "\\\\" + nomFichierIn));		
		}
		
	}
	///////////////////////////// fonctions detect() /////////////////////////////////////////////////////////////////////:
	
	//--- fonction principale detect() : fonction de détection - renvoie le tableau des Rectangles
	public Rectangle[] detect( opencv_core.IplImage iplImgIn, float scaleIn, int min_neighborsIn, int flagsIn, int min_widthIn, int min_heightIn, int max_widthIn, int max_heightIn, boolean debugIn  ){
	  
	  //--- objets utiles pour les rectangles détectés --- 	  
	  // alternative à un tableau fixe = créer un ArrayList de Rectangle (une sorte de tableau à taille variable) 
	  //-- le Arraylist est déclaré en static 
	 myRectList = new ArrayList<Rectangle>(); // réinitialise le Array List static 

		
	  //--- objets utiles et paramètres pour la détection ---   
	  opencv_core.CvMemStorage storage= opencv_core.cvCreateMemStorage(0); // initialise objet conteneur CvMemStorage 
	  //float scaleIn=1.2; // facteur d'échelle = 1.1 par défaut
	  //int neighbors=2; // 0 : tous les rectangle - 3 par défaut - 2 conseillé
	  //int flags=opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING; // d'autres possibles... cf  notamment opencv_objdetect : CV_HAAR_FIND_BIGGEST_OBJECT 

	  //--- récupère la séquence d'objets détectés --- 
	  
	  // static opencv_core.CvSeq cvHaarDetectObjects(opencv_core.CvArr image, opencv_objdetect.CvHaarClassifierCascade cascade, opencv_core.CvMemStorage storage, double scale_factor, int min_neighbors, int flags, opencv_core.CvSize min_size, opencv_core.CvSize max_size) 
	  opencv_core.CvSeq faces = opencv_objdetect.cvHaarDetectObjects( iplImgIn, cascade,storage,scaleIn, min_neighborsIn,flagsIn,opencv_core.cvSize( min_widthIn, min_heightIn ), opencv_core.cvSize( max_widthIn, max_heightIn ) );
	  // nb : minnimum size = 20x20 car c'est la taille des images utilisées pour le training du fichier de reconnaissance
	  
	  if (debugIn) PApplet.println ("Nombre objets détectés =" + faces.total()); 

	  // récupère les rectangles à partir de la séquence 
	  opencv_core.CvRect myCvRect; // déclare objet CvRect 
	  
	  for( int i=0; i<faces.total(); i++ ) { // parcourt les éléments de la séquence - la séquence contient des CvRect 

		myCvRect = new opencv_core.CvRect(opencv_core.cvGetSeqElem( faces, i )); // récupère l'objet CvRect courant de la séquence
	        // crée un nouvel objet en se basant sur le pointeur renvoyé par la fonction cvGetSeqElem(seq, indice) - cette fonction renvoie un javacpp.Pointer et non objet

		
	        //--- affiche infos sur le rectangle -- 
		 if (debugIn) PApplet.println ("Rectangle " + i +" : x="+ myCvRect.x()+" | y="+ myCvRect.y()+" | w="+ myCvRect.width()+" | h="+ myCvRect.height()); 
	        
	    /*
	        //-- dessine le rectangle -cf drawRectDetect()
	        p.noFill();
	        p.stroke(p.color(255,0,0));
	        p.rect(myCvRect.x(),myCvRect.y(),myCvRect.width(),myCvRect.height());   
	     */
		
	     //--- ajoute le rectangle à la liste ---
	     myRectList.add(new Rectangle(myCvRect.x(),myCvRect.y(),myCvRect.width(),myCvRect.height()));
				   
	        		
	  } // fin for i 
	  
	  Rectangle[] myRects = (Rectangle[]) myRectList.toArray(new Rectangle[0]);// récupère le ArrayList dans le tableau de Rectangle
	  
	  //storage.release(); // libère la mémoire - pb ?
	  
	  
	  return(myRects); 
	} // fin fonction detect() principale 
	
	//fonction detect() minimale
	public Rectangle[] detect() {
		
		return (detect( Buffer, (float)1.2, 2, opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING, p.width/16, p.height/16, p.width, p.height, false  ));
	}
	

	//fonction detect() minimale debug
	public Rectangle[] detect(boolean debugIn) {
		
		return (detect( Buffer, (float)1.2, 2, opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING, p.width/16, p.height/16, p.width, p.height, debugIn  ));
	}

	//fonction detect() recevant iplImage
	public Rectangle[] detect(opencv_core.IplImage iplImgIn) {
		
		return (detect( iplImgIn, (float)1.2, 2, opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING, p.width/16, p.height/16, p.width, p.height, false  ));
	}

	//fonction detect() recevant iplImage et debug 
	public Rectangle[] detect(opencv_core.IplImage iplImgIn, boolean debugIn) {
		
		return (detect( iplImgIn, (float)1.2, 2, opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING, p.width/16, p.height/16, p.width, p.height, debugIn  ));
	}
	
	////--- fin fonctions detect -------------------
	
	//////////// fonction générale drawRects() //////////
	
	// fonction principale drawRects : trace le rectangle  de tous les rectangles d'un tableau de rectangles  
	public void drawRects (Rectangle[] rectsIn, int xRefIn, int yRefIn, float scaleIn, int colorStrokeIn, int strokeWeightIn, boolean fillIn, int colorFillIn, boolean debugIn) {
		
		// la fonction reçoit : 
		// le tableau de Rectangle utile
		// les coordonnées de référence pour le tracé (coin sup gauche)
		// le facteur d'échelle éventuel à appliquer - mettre à 1 par défaut 
		// la couleur du pourtour
		// si remplissage
		// couleur remplissage
		// épaisseur pourtour
		// le flag de debug
		
		
		if (rectsIn!=null) { // si le tableau de Blob n'est pas vide 
			

		      for( int i=0; i<rectsIn.length; i++ ) { // passe en revue les blobs (= formes détectées)

		          //---- Récupération des paramètres du rectangle entourant l'objet courant ---- 
	              Rectangle rectangleDetect=rectsIn[i]; // récupère le rectangle qui contient la forme détectée
		          
		          
		          //---------- fixe les paramètres graphiques  -----------
		          if (fillIn) p.fill(colorFillIn) ; else p.noFill(); 
		          p.stroke(colorStrokeIn); // couleur verte
		          p.strokeWeight(strokeWeightIn); // épaisseur pourtour 
		          
			        //--- affiche infos sur le rectangle -- 
		 		 if (debugIn) PApplet.println ("Rectangle " + i +" : x="+ rectangleDetect.x+" | y="+ rectangleDetect.y+" | w="+ rectangleDetect.width+" | h="+ rectangleDetect.height); 

		         //--- tracé du rectangle entourant le blob courant  
		         p.rect( xRefIn+(rectangleDetect.x*scaleIn), yRefIn+(rectangleDetect.y*scaleIn), rectangleDetect.width*scaleIn, rectangleDetect.height*scaleIn );

		      } // fin for 
		      
		} // fin if (blobsIn!=null)
	
	      
	} // fin drawRectBlobs principale 
	
	//////////// fonctions drawRectDetect() //////////
	// basée sur la fonction générale drawRects
	
	// fonction drawRectDetect par défaut ...
		public void drawRectDetect() {

		  Rectangle[] myRects = (Rectangle[]) myRectList.toArray(new Rectangle[0]);// récupère le ArrayList static dans le tableau de Rectangle
		  
		  drawRects (myRects,0, 0, 1, p.color(255,0,0), 1 , false, 0, false); // dessin rectangle par défaut

	} // fin drawRectDetect

		// fonction drawRectDetect par défaut avec debug...
		public void drawRectDetect(boolean debugIn) {

		  Rectangle[] myRects = (Rectangle[]) myRectList.toArray(new Rectangle[0]);// récupère le ArrayList static dans le tableau de Rectangle
		  
		  drawRects (myRects,0, 0, 1, p.color(255,0,0), 1 , false, 0, debugIn); // dessin rectangle par défaut

	} // fin drawRectDetect
		
		// fonction drawRectDetect complète
		public void drawRectDetect(int xRefIn, int yRefIn, float scaleIn, int colorStrokeIn, int strokeWeightIn, boolean fillIn, int colorFillIn, boolean debugIn) {

			  Rectangle[] myRects = (Rectangle[]) myRectList.toArray(new Rectangle[0]);// récupère le ArrayList static dans le tableau de Rectangle
			  
			  drawRects (myRects,xRefIn, yRefIn, scaleIn, colorStrokeIn, strokeWeightIn , fillIn, colorFillIn, debugIn); // dessin rectangle par défaut

		} // fin drawRectDetect


		
		//////////// fonction générale drawCircle  //////////
		
		// fonction principale drawCircle : trace un cercle   à partir point et rayon 
		public void drawCircle (
				Point center, // le centre du cercle à tracer
				int xRefIn, int yRefIn, // les coordonnées du cercle à tracer
				float scaleIn, // l'échelle à utiliser
				int radius, // rayon à utiliser 
				int colorStrokeIn, int strokeWeightIn, // couleur et épaisseur du pourtour du cercle
				boolean fillIn, int colorFillIn, // drapeau de remplissage et couleur de remplissage
				boolean debugIn // drapeau d'affichage des messages 
				)
		{
			
			          
			          //---------- fixe les paramètres graphiques  -----------
			          if (fillIn) p.fill(colorFillIn) ; else p.noFill(); 
			          p.stroke(colorStrokeIn); // couleur verte
			          p.strokeWeight(strokeWeightIn); // épaisseur pourtour 
			          
				        //--- affiche infos sur le cercle-- 
			 		 if (debugIn) PApplet.println ("Trace cercle de centre : x="+ center.x+" | y="+ center.y + " et de rayon = " + radius); 
			 		 
			          //---- dessine cercle 
			          p.ellipse(xRefIn+(center.x*scaleIn), yRefIn+(center.y*scaleIn), radius,radius); 
		
		      
		} // fin drawCircle  principale 
		
		// fonction similaire - drawPoint() = appelle drawCircle()  => utiliser ellipse plutôt... ? 
		public void drawPoint (				
				Point center, // le centre du cercle à tracer
				int xRefIn, int yRefIn, // les coordonnées du cercle à tracer
				float scaleIn, // l'échelle à utiliser
				int radius, // rayon à utiliser 
				int colorStrokeIn, int strokeWeightIn, // couleur et épaisseur du pourtour du cercle
				boolean fillIn, int colorFillIn, // drapeau de remplissage et couleur de remplissage
				boolean debugIn // drapeau d'affichage des messages 
			) 
		{
			
			drawCircle(
					center, // le centre du cercle à tracer
					xRefIn, yRefIn, // les coordonnées du cercle à tracer
					scaleIn, // l'échelle à utiliser
					radius, // rayon à utiliser 
					colorStrokeIn, strokeWeightIn, // couleur et épaisseur du pourtour du cercle
					fillIn, colorFillIn, // drapeau de remplissage et couleur de remplissage
					debugIn // drapeau d'affichage des messages 
				); 
			
		}

		// fonction principale drawCircle variante : trace un cercle   à partir d'un objet Circle
		public void drawCircle (
				Circle circleIn, // l'objet Circle à utiliser
				int xRefIn, int yRefIn, // les coordonnées du cercle à tracer
				float scaleIn, // l'échelle à utiliser
				float radiusScaleIn, // l'échelle à utiliser pour rayon				
				int colorStrokeIn, int strokeWeightIn, // couleur et épaisseur du pourtour du cercle
				boolean fillIn, int colorFillIn, // drapeau de remplissage et couleur de remplissage
				boolean debugIn // drapeau d'affichage des messages 
				)
		{
			
			
			          //---------- fixe les paramètres graphiques  -----------
			          if (fillIn) p.fill(colorFillIn) ; else p.noFill(); 
			          p.stroke(colorStrokeIn); // couleur verte
			          p.strokeWeight(strokeWeightIn); // épaisseur pourtour 
			          
				        //--- affiche infos sur le cercle-- 
			 		 if (debugIn) PApplet.println ("Trace cercle de centre : x="+ circleIn.center.x+" | y="+ circleIn.center.y+ " et de rayon = " + circleIn.radius); 
			 		 
			          //---- dessine cercle 
			          p.ellipse(xRefIn+(circleIn.center.x*scaleIn), yRefIn+(circleIn.center.y*scaleIn), circleIn.radius*scaleIn*radiusScaleIn,circleIn.radius*scaleIn*radiusScaleIn); 
		

		
		      
		} // fin drawCircle  principale - variante à partir objet Circle 

		// ------- fonction principale drawCenterCircle  ----- : trace le centre d'un objet Circle 
		public void drawCenterCircle (
				Circle circleIn, // l'objet Circle à utiliser
				int xRefIn, int yRefIn, // les coordonnées du cercle à tracer
				float scaleIn, // l'échelle à utiliser
				int radius, // rayon à utiliser 
				int colorStrokeIn, int strokeWeightIn, // couleur et épaisseur du pourtour du cercle
				boolean fillIn, int colorFillIn, // drapeau de remplissage et couleur de remplissage
				boolean debugIn // drapeau d'affichage des messages 
				)
		{
			
	          //---------- fixe les paramètres graphiques à utiliser -----------
	          if (fillIn) p.fill(colorFillIn) ; else p.noFill(); 
	          p.stroke(colorStrokeIn); // couleur verte
	          p.strokeWeight(strokeWeightIn); // épaisseur pourtour 

	          //---------- dessine un cercle autour du centre -----------
	          p.ellipse (xRefIn+(circleIn.center.x*scaleIn),yRefIn+(circleIn.center.y*scaleIn), radius,radius);

			
		} // fin drawCenterCircle

		
		// fonction principale drawCircles : trace un tableau d'objet Circle 
		public void drawCircles (
				Circle[] circlesIn, // le tableau de cercles à tracer
				int xRefIn, int yRefIn, // les coordonnées du cercle à tracer
				float scaleIn, // l'échelle du dessin à utiliser
				float radiusScaleIn, // l'échelle à utiliser pour les rayons
				int colorStrokeIn, int strokeWeightIn, // couleur et épaisseur du pourtour du cercle
				boolean fillIn, int colorFillIn, // drapeau de remplissage et couleur de remplissage
				boolean debugIn // drapeau d'affichage des messages 
				)
		{
			
			if (circlesIn!=null) { // si le tableau n'est pas vide 
				
				for (int i=0; i<circlesIn.length; i++ ) { // défile les éléments du tableau 
					
					if (debugIn) PApplet.println(" Cercle " + i + " : " ); 
					drawCircle (circlesIn[i], xRefIn, yRefIn, scaleIn, radiusScaleIn, colorStrokeIn, strokeWeightIn, fillIn, colorFillIn, debugIn); // dessine le cercle 
					
				} // fin for 
				
			}// fin if circlesIn != null
			
		} // fin draw Circles 
		
		// fonction principale drawCenterCircles : trace les centres d'un tableau d'objet Circle 
		public void drawCenterCircles (
				Circle[] circlesIn, // le tableau de cercles à tracer
				int xRefIn, int yRefIn, // les coordonnées du cercle à tracer
				float scaleIn, // l'échelle à utiliser
				int radiusIn, // rayon à utiliser 
				int colorStrokeIn, int strokeWeightIn, // couleur et épaisseur du pourtour du cercle
				boolean fillIn, int colorFillIn, // drapeau de remplissage et couleur de remplissage
				boolean debugIn // drapeau d'affichage des messages 
				)
		{
			
			if (circlesIn!=null) { // si le tableau n'est pas vide 
				
				for (int i=0; i<circlesIn.length; i++ ) { // défile les éléments du tableau 
					
					if (debugIn) PApplet.println(" Cercle " + i + " : " ); 
					drawCenterCircle (circlesIn[i], xRefIn, yRefIn, scaleIn, radiusIn,colorStrokeIn, strokeWeightIn, fillIn, colorFillIn, debugIn); // dessine le cercle 
					
				} // fin for 
				
			}// fin if circlesIn != null
			
		} // fin drawCenterCircles 
		
		//////// fonctions de soustraction du fond ///////////////
		
		// cf absDiff
		
		/////////////// MOG //////////////////////
		
		//--- fonction bgsMOGInit pour BackgroundSubstract MOG (Mixture of Gaussian) 
		
		public void bgsMOGInit(int history, int nmixtures, double backgroundRatio, double noiseSigma) {
			
			// cette fonction implémente l'algorithme natif Opencv MOG pour la soustraction du fond 
			//--- initialise un objet opencv_video.BackgroundSubtractorMOG
			
			// Cette fonction reçoit : 
			// history : nombre de frame à mémoriser pour l'histoire du background Model
			// nmixture : le nombre de gaussiennes à utiliser
			// backgroundRatio : seuillage
			// noiseSigma : coeff bruit
			
			// voir : http://opencv.itseez.com/modules/video/doc/motion_analysis_and_object_tracking.html#backgroundsubtractormog
			
			

	        //opencv_video.BackgroundSubtractorMOG(int history, int nmixtures, double backgroundRatio, double noiseSigma) 
	        //bgsMOG=new opencv_video.BackgroundSubtractorMOG(200, 1, 0.1, 10); // initialise bgsMOG avec paramètres voulus 
			bgsMOG=new opencv_video.BackgroundSubtractorMOG(history, nmixtures, backgroundRatio, noiseSigma); // initialise bgsMOG avec paramètres voulus
			
			//nb : bgsMog est déclaré en tant qu'objet global de la classe Opencv
		}

		//-- forme simple ---
		public void bgsMOGInit() {

	        bgsMOG=new opencv_video.BackgroundSubtractorMOG(); // initialise bgsMOG avec paramètres par défaut
	        // default : history = 200; nmixture=5; backgroundRatio (threshold) = 0.7; noiseSigma=15; 
	
		}

		//--- fonction bgsMOGApply : ajoute une image au background Model : typiquement une frame 
		
		public void bgsMOGApply(opencv_core.IplImage iplImgSrcIn, opencv_core.IplImage iplImgDestIn, int modeIn) {
			
			// la fonction reçoit : 
			// objet IplImage à ajouter
			// objet IplImage destination
			// mode de fonctionnnement de l'algorithme - 0 ou -1.. 
			
			bgsMOG.apply(iplImgSrcIn, iplImgDestIn, modeIn); // ajoute un frame à l'objet BackGround Subtractor
				
			//nb : bgsMog est déclaré en tant qu'objet global de la classe Opencv
		}

		//--- fonction bgsMOGApply :forme simple
		
		public void bgsMOGApply() {
			
			// la fonction reçoit : 
			// objet IplImage à ajouter
			// objet IplImage destination
			// mode de fonctionnnement de l'algorithme - 0 ou -1.. 
			
			bgsMOG.apply(Buffer, BufferGray, 0); // ajoute un frame à l'objet BackGround Subtractor
				
			//nb : bgsMog est déclaré en tant qu'objet global de la classe Opencv
		}

		/////////////// MOG2 //////////////////////
		
		//--- fonction bgsMOG2Init pour BackgroundSubstract MOG2 (Mixture of Gaussian) 
		
		public void bgsMOG2Init(int history, float varThreshold, boolean bShadowDetection) {
			
			// cette fonction implémente l'algorithme natif Opencv MOG2 pour la soustraction du fond 
			//--- initialise un objet opencv_video.BackgroundSubtractorMOG
			
			// Cette fonction reçoit : 
			// history : nombre de frame à mémoriser pour l'histoire du background Model
			// varThreshold : seuillage
			// bShadowDetection : drapeau détection de l'ombre
			
			// voir : http://opencv.itseez.com/modules/video/doc/motion_analysis_and_object_tracking.html#backgroundsubtractormog2
			

		       // opencv_video.BackgroundSubtractorMOG2(int history, float varThreshold, boolean bShadowDetection) - nb shadow = ombre
		       //bgsMOG2=new opencv_video.BackgroundSubtractorMOG2(100,10,true); // initialise bgsMOG avec paramètres 
			bgsMOG2=new opencv_video.BackgroundSubtractorMOG2(history, varThreshold,bShadowDetection); // initialise bgsMOG2 avec paramètres voulus
			
			//nb : bgsMog2 est déclaré en tant qu'objet global de la classe Opencv
		}

		//-- forme simple ---
		public void bgsMOG2Init() {

	        bgsMOG2=new opencv_video.BackgroundSubtractorMOG2(); // initialise bgsMOG avec paramètres par défaut
	        // default : 
	        
	        //PApplet.println("History ="+ bgsMOG2.history()); 
	        //PApplet.println("varThreshold ="+ bgsMOG2.varThreshold()); 
	        //PApplet.println("bShadowDetection ="+ bgsMOG2.bShadowDetection()); 
	    	
		}

		//--- fonction bgsMOG2Apply : ajoute une image au background Model : typiquement une frame 
		
		public void bgsMOG2Apply(opencv_core.IplImage iplImgSrcIn, opencv_core.IplImage iplImgDestIn, int modeIn) {
			
			// la fonction reçoit : 
			// objet IplImage à ajouter
			// objet IplImage destination
			// mode de fonctionnnement de l'algorithme - 0 ou -1.. Utiliser -1
			
			bgsMOG2.apply(iplImgSrcIn, iplImgDestIn, modeIn); // ajoute un frame à l'objet BackGround Subtractor
				
			//nb : bgsMog2 est déclaré en tant qu'objet global de la classe Opencv
		}

		//--- fonction bgsMOG2Apply :forme simple
		
		public void bgsMOG2Apply() {
			
			// la fonction reçoit : 
			// objet IplImage à ajouter
			// objet IplImage destination
			// mode de fonctionnnement de l'algorithme - 0 ou -1.. 
			
			bgsMOG2.apply(Buffer, BufferGray, -1); // ajoute un frame à l'objet BackGround Subtractor
				
			//nb : bgsMog2 est déclaré en tant qu'objet global de la classe Opencv
		}
		///////////////// fonctions math / géométriques utiles  intégrées à la librairie ////////////////////::
		
		/*
		 * Ces fonctions mathématiques sont intégrées à la librairie en raison de leur praticité potentielle
		 * Elles peuvent être basées sur les fonctions natives openCV ou non... 
		 * Elles sont utilisée par certaines fonctions de la librairie
		 */
		
		//----------- calcul de la distance entre 2 points ----- 
		public float distance( Point startIn, Point endIn) {
			
			float distanceOut=0;
			float calcX=0; 
			float calcY=0; 
			
			
			calcX=PApplet.pow((endIn.x-startIn.x),2);
			calcY=PApplet.pow((endIn.y-startIn.y),2);
			
			distanceOut=PApplet.sqrt( calcX  + calcY ); // avec fonction math Processing
			
			return (distanceOut); 
		}
		
		//----------- calcul les paramètres a et b de l'équation  d'une droite y=ax+b à partir de 2 points ---------- 
		public float[] calculEquationDroite( Point pointA, Point pointB) {
			// renvoie un tableau de 2 float tel que [0] vaut a et [1] vaut b 
			
			  // info utiles : http://www.ahristov.com/tutorial/geometry-games/lines-2d.html
			 
			float xA=pointA.x; 
			float yA=pointA.y;
			
			float xB=pointB.x; 
			float yB=pointB.y; 
		
			//----- calcul des paramètres a et b de la droite y=ax+b
			 float a= (yB-yA)/(xB-xA); 
			  
			 float b= yB - (a * xB); 

			 float[] abOut=new float[2]; 
			 abOut[0]=a; 
			 abOut[1]=b;
			 
			return (abOut); 
			
		} // fin calcul pente		

		//------------ interLines : renvoie le point d'intersection de 2 droites à partir des paramètres a et b des 2 droites----- 
		public Point interLines( float a1, float b1, float a2, float b2) {
			  
			  // infos utiles ici : http://www.ahristov.com/tutorial/geometry-games/intersection-lines.html
			  
			  // droite 1 : y = a1 . x + b1
			  // droite 2 : y = a2 . x + b2
			  
			  // on a: 
			  // x = (b2-b1)/(a1-a2)
			  // y = a1x + b1 = a1(b2-b1)/(a1-a2) + b1
			
			  PApplet.println (" a1= " + a1 + "| b1=" + b1 + "| a2="+ a2 + "| b2=" + b2); 
			  
			  //-- calcul x 
			  float x= (b2-b1) / (a1-a2); 
			  PApplet.println (" x= " + x); 

			  //-- calcul y 
			  float y= (a1* x) + b1; 
			  PApplet.println (" y= " + y); 
			  
			  //--- renvoi de la fonction 
			  Point pointOut= new Point(); 
			  
			  pointOut.x=(int)x; 
			  pointOut.y=(int)y; 
			  
			  return(pointOut); // appelle fonction interLines
			    
			} // fin intersectionDroites à partir 2 objets Line
		
		//------------ interLines : renvoie le point d'intersection de 2 droites à partir de 2 objets Line ----- 
		public Point interLines(Line line1In, Line line2In ) {
			  
			  // infos utiles ici : http://www.ahristov.com/tutorial/geometry-games/intersection-lines.html
			  
			  // droite 1 : y = a1 . x + b1
			  // droite 2 : y = a2 . x + b2
			  
			  // on a: 
			  // x = (b2-b1)/(a1-a2)
			  // y = a1x + b1 = a1(b2-b1)/(a1-a2) + b1
			
			// --- Line 1 
			float a1 = line1In.a; 
			float b1=line1In.b; 
			
			//--- Line 2 
			float a2=line2In.a; 
			float b2=line2In.b;
			  
			  return(interLines( a1, b1,  a2, b2)); 
			    
			} // fin intersectionDroites
		
		//------------ fonction théorème de Al-Kashi pour calcul angle à partir 3 côtés ---------------------

		/*

		Cette fonction renvoie la valeur d'un angle d'un triangle quelconque connaissant
		la longueur des 3 côtés du triangle, à savoir des 2 cotés adjacents et du coté opposé à cet angle.

		Cette fonction applique le theoreme de Al-Kashi (ou théorème de Pythagore élargi)
		Voir notamment : http://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me_d%27Al-Kashi

		Formule du theoreme de Al-Kashi :

		                  (adj1² + adj2² - opp²)
		angle =  acos   ( ------------------------- )
		                  (2 x adj1 x adj2)

		*/

		public float calculAngleRadAlKashi( float adj1, float adj2, float opp) {
		// avec :
		// adj1 et adj2 la longueur des 2 cotés adjacents à l'angle à calculer
		//opp la longueur du coté opposé à l'angle à calculer

		 //----------- variables utilisées par la fonction
		 float D; // variable calcul dénominateur
		 float N;  // variable calcul numérateur
		 float calculAngleRad; // variable angle à calculer

		 //----------- calcul du dénominateur : (adj1² + adj2² - opp²) ---------

		 D=PApplet.pow(adj1,2) + PApplet.pow (adj2,2) - PApplet.pow(opp,2);  // basé sur fonction math Processing


		 //----------- calcul du numérateur :  (2 x adj1 x adj2) ---------------

		 N=2*adj1*adj2;

		 //-------------- calcul final de l'angle ----------------------------

		 calculAngleRad=PApplet.acos(D/N); // calcule l'angle en radians  - basé sur fonction math Processing

		 //----- renvoi de la valeur calculée ----
		 return (calculAngleRad);

		}


		//---------------------- fin fonction théorème Al-Kashi calcul angle à partir 3 cotés ------------------

		//------------ fonction théorème de Al-Kashi pour calcul 3eme coté à partir 2 côtés et un angle (en radian) ---------------------

		/*

		Cette fonction renvoie la valeur du 3ème côté d'un triangle quelconque connaissant
		l'angle opposé à ce 3ème côté
		et la longueur des 2 autres côtés du triangle, à savoir des 2 cotés adjacents à l'angle utilisé.

		Cette fonction applique le theoreme de Al-Kashi (ou théorème de Pythagore élargi)
		Voir notamment : http://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me_d%27Al-Kashi

		Formule du theoreme de Al-Kashi :

		opp=sqrt(adj1² + adj2² - (2 x adj1 x adj2 x cos (angle)))

		*/

		public float calculCoteAlKashi( float adj1, float adj2, float angleOpp) {
		// avec :
		// angle est l'angle opposé au 3ème coté
		// adj1 et adj2 la longueur des 2 cotés adjacents à l'angle utilisé

		 //----------- variables utilisées par la fonction
		 float D1; // variable calcul partiel
		 float D2;  // variable calcul partiel
		 float calculCote; // variable 3ème Coté à caclculer

		 //----------- calcul partiel  adj1² + adj2² ---------

		 D1=PApplet.pow(adj1,2) + PApplet.pow (adj2,2); // basé sur fonction math Processing


		 //----------- calcul partiel  (2 x adj1 x adj2 x cos (angle)) ---------------

		 D2=2*adj1*adj2*PApplet.cos(angleOpp); // basé sur fonction math Processing

		 //-------------- calcul final du coté ----------------------------

		 calculCote=PApplet.sqrt(D1-D2); // calcule le 3ème coté - basé sur fonction math Processing

		 //----- renvoi de la valeur calculée ----
		 return (calculCote);

		}


		//---------------------- fin fonction théorème Al-Kashi pour calcul 3ème coté ------------------

		
		//----- meanPoints() : fonction recevant un tableau de Points et renvoyant un point correspondant à la moyenne des coordonnées de l'ensemble des points
		
		public Point meanPoints(Point[] pointsIn, boolean debug){
		
			Point pointOut= new Point(); 
			
			if (pointsIn.length>0) { // si au moins un élément dans le tableau 
				
			    // création d'un CvMat de stockage des points In  : ngroupes de 2 valeurs = 1 ligne x 4 colonnes x 2 canaux 
			    opencv_core.CvMat cvMatPointsIn= opencv_core.CvMat.create(1,pointsIn.length,opencv_core.CV_32F,2); // crée un CvMat de 4 points = 1 ligne x 4 colonnes - 32F - 2 canaux
	
			    //--- met les coordonnées des pointIn dans le CvMat 
			    
			     if(debug) PApplet.println ("----- Coordonnées des points ---------- "); 

				   //--- affiche le contenu du CVMat --- 
				   for (int n=0; n<pointsIn.length; n++) { // défile les points 

					    cvMatPointsIn.put(2*n,pointsIn[n].x) ; // met la valeur x à l'index 2n 
					    cvMatPointsIn.put((2*n)+1,pointsIn[n].y) ; // met la valeur x à l'index 2n 

					
				     if(debug) PApplet.println ("Point " + n + " :  x= "+ cvMatPointsIn.get(2*n)+" | y= " + cvMatPointsIn.get((2*n)+1)); 
				     
				   } // fin for 
					
				   
				   //------- calcul de la moyenne des points... 
				   
				   //opencv_core.CvScalar cvAvg(opencv_core.CvArr arr, opencv_core.CvArr mask)
				   
				   opencv_core.CvScalar myScalar=opencv_core.cvAvg(cvMatPointsIn, null);
				  
				   if (debug) PApplet.println("x mean = " + myScalar.getVal(0));
				   if (debug) PApplet.println("y mean = " + myScalar.getVal(1));
				   
				   //--- récupère les coordonnées moyenne dans le point à renvoyer 
				   pointOut.x=(int)myScalar.getVal(0);
				   pointOut.y=(int)myScalar.getVal(1);
				   
			}
			
			else {
				
				PApplet.println("Tableau de points reçu vide !"); 
			}

			return(pointOut);
		
		}

		///////////////// fonctions d'analyse Feature 2D - SURF, SimpleBlobDetector, etc..  ////////////////////::
		
		//============ fonction communes ========================
		public void drawKeypoints (
				Keypoint[] keypointsIn, // tableau de keypoints à dessiner
				int xRefIn, int yRefIn, float scaleIn, // coordonnées de référence pour le dessin et échelle
				int radius, // rayon des points - mettre -1 si utilisation size des Keypoints
				int colorStrokeIn, int strokeWeightIn, // couleur et épaisseur du pourtour
				boolean fillIn, int colorFillIn, // drapeau et couleur de remplissage
				boolean debug // drapeau pour affichage des messages
				)
				{
		
			
			  if (debug) PApplet.println ("Taille vecteur de points-clés objet ="+ keypointsIn.length); 
			  
			  for (int i=0; i<keypointsIn.length; i++) { // défile les point-clés unitaires au sein du vecteur de Keypoint
			    
			    
			    if (debug) PApplet.print("Point clé "+ i +" : x="+keypointsIn[i].point.x+ " | y="+keypointsIn[i].point.y); // coordonnées du point
			    if (debug) PApplet.print( " | taille="+keypointsIn[i].size); // taille 
			    if (debug) PApplet.print( " | angle="+keypointsIn[i].angle); // angle
			    if (debug) PApplet.println ( " | octave="+keypointsIn[i].octave); // octave
			    
			    
		          //---------- fixe les paramètres graphiques à utiliser -----------
		          if (fillIn) p.fill(colorFillIn) ; else p.noFill(); 
		          p.stroke(colorStrokeIn); // couleur verte
		          p.strokeWeight(strokeWeightIn); // épaisseur pourtour 

		          //--- tracé des point 
		          if (radius==-1){ // si dessin des cercle en utilisant le paramètre size des keypoints
		        	  p.ellipse (xRefIn+(scaleIn*keypointsIn[i].point.x), yRefIn+(scaleIn*keypointsIn[i].point.y), keypointsIn[i].size*scaleIn,keypointsIn[i].size*scaleIn);
		          }
		          else { // sinon utilise la valeur propre de radius
		        	  p.ellipse (xRefIn+(scaleIn*keypointsIn[i].point.x), yRefIn+(scaleIn*keypointsIn[i].point.y), radius*scaleIn,radius*scaleIn);		        	  
		          }
			    
			  } // fin for 

			
			
			
		} // fin drawKeypoints complète 
		
		
		//--- forme minimale
		public void drawKeypoints (Keypoint[] keypointsIn, boolean debug) {
			
			drawKeypoints ( 
					keypointsIn, //Keypoint[] keypointsIn, 
					0,0,1,//int xRefIn, int yRefIn, float scaleIn, 
					-1, // int radius - utiliser -1 pour tracer cercle avec valeur de size des Keypoints
					p.color(255,0,255), 2, //int colorStrokeIn, int strokeWeightIn, 
					false,0,//boolean fillIn, int colorFillIn, 
					debug//boolean debug
					);

			
		} // fin drawkeypoints
		
		
		//============ keypointsToPoints() = reçoit un tableau de Keypoints et renvoie un tableau de Points 
		
		public Point[] keypointsToPoints(Keypoint[] keypointsIn) {
			
			Point[] pointsOut = new Point[keypointsIn.length]; 
			
			for (int i=0; i<keypointsIn.length; i++) {
				
				pointsOut[i]=new Point(keypointsIn[i].point.x, keypointsIn[i].point.y); // récupère les coordonnées du point du Keypoint(i) dans le point(i)
				
			} // fin for 
						
			return (pointsOut); 
		}
		
		
		//=========== SimpleBlobDetector ========================
		
		//------- fonction principale keypointsSBD = keypoints Simple Blob Detector 
		public Keypoint[] keypointsSBD (
				opencv_core.IplImage iplImgSceneIn,  
				float minThreshold, float maxThreshold, float thresholdStep,
				float minDistBetweenBlobs,
				boolean filterByColor, int blobColor,
				boolean filterByArea, float minArea, float maxArea, 
				boolean filterByCircularity, float minCircularity, float maxCircularity,
				boolean filterByConvexity, float minConvexity, float maxConvexity,
				boolean filterByInertia, float minInertiaRatio, float maxInertiaRatio,
				long minRepeatability,
				boolean debug) { // la fonction Blobs renvoie un tableau de blobs

			myKeypointsListScene = new ArrayList<Keypoint>(); // alternative à un tableau fixe = créer un ArrayList de Keypoint  - déclaré ici en static - cf entête classe 

			//--- les éléments du keypoint courant ---
		    Point  pointKeypoint=new Point(); // point du Keypoint
		    float sizeKeypoint; // rayon du point clé
		    float angleKeypoint; // angle du point clé
		    int octaveKeypoint; //octave du point clé
		    

			
			  //--- Etape 0 : initialisation du SimpleBlobDetector

			   // -- initilisation de opencv_features2d.SimpleBlobDetector.Params
			   opencv_features2d.SimpleBlobDetector.Params parameters = new opencv_features2d.SimpleBlobDetector.Params(); 

			   //-- +/- définition des paramètres utilisés --- 

			   //-- paramétrage par défaut --
			    parameters.maxThreshold(maxThreshold);// float 	maxThreshold()
			    parameters.minThreshold(minThreshold);// float 	minThreshold()
			    parameters.thresholdStep(thresholdStep);// float 	thresholdStep()
			   
			   
			    parameters.minDistBetweenBlobs(minDistBetweenBlobs);//float 	minDistBetweenBlobs()
			    
			    parameters.filterByColor(filterByColor);//boolean 	filterByColor()
			    parameters.blobColor((byte)blobColor);//byte 	blobColor() - en niveau de gris 0-255
			    
			    parameters.filterByArea(filterByArea);//boolean 	filterByArea()
			    parameters.maxArea(maxArea);//float 	maxArea()
			    parameters.minArea(minArea);//float 	minArea()
			    
			    //--- la circularité calcule 4*pi*area / perimetre² --- 
			    // dans le cas du cercle : périmètre = 2 x pi x r et aire = pi * r²
			    // la circularité sera 4 * pi * pi * r² / (2 * pi * r) ² = 4 * pi² * r² / 4 * pi² * r² =1
			     parameters.filterByCircularity(filterByCircularity);//boolean 	filterByCircularity()
			    parameters.maxCircularity(maxCircularity);//float 	maxCircularity()
			    parameters.minCircularity(minCircularity);//float 	minCircularity()
			    
			     parameters.filterByConvexity(filterByConvexity);//boolean 	filterByConvexity()
			    parameters.maxConvexity(maxConvexity);//float 	maxConvexity()
			    parameters.minConvexity(minConvexity);//float 	minConvexity()
			    
			    parameters.filterByInertia(filterByInertia);//boolean 	filterByInertia()
			    parameters.maxInertiaRatio(maxInertiaRatio);//float 	maxInertiaRatio()
			    parameters.minInertiaRatio(minInertiaRatio);//float 	minInertiaRatio()
			    
			    parameters.minRepeatability(minRepeatability);//long 	minRepeatability() ?

			   /*
			   //-- paramétrage par défaut --
			    parameters.maxThreshold(250);// float 	maxThreshold()
			    parameters.minThreshold(150);// float 	minThreshold()
			    parameters.thresholdStep(10);// float 	thresholdStep()
			   
			   
			    parameters.minDistBetweenBlobs(1);//float 	minDistBetweenBlobs()
			    
			    parameters.filterByColor(false);//boolean 	filterByColor()
			    parameters.blobColor((byte)255);//byte 	blobColor() - en niveau de gris 0-255
			    
			    parameters.filterByArea(false);//boolean 	filterByArea()
			    parameters.maxArea(5000);//float 	maxArea()
			    parameters.minArea(250);//float 	minArea()
			    
			    //--- la circularité calcule 4*pi*area / perimetre² --- 
			    // dans le cas du cercle : périmètre = 2 x pi x r et aire = pi * r²
			    // la circularité sera 4 * pi * pi * r² / (2 * pi * r) ² = 4 * pi² * r² / 4 * pi² * r² =1
			     parameters.filterByCircularity(false);//boolean 	filterByCircularity()
			    parameters.maxCircularity((float)1.2);//float 	maxCircularity()
			    parameters.minCircularity((float)0.8);//float 	minCircularity()
			    
			     parameters.filterByConvexity(false);//boolean 	filterByConvexity()
			    //parameters.maxConvexity();//float 	maxConvexity()
			    //parameters.minConvexity();//float 	minConvexity()
			    
			    parameters.filterByInertia(false);//boolean 	filterByInertia()
			    //parameters.maxInertiaRatio();//float 	maxInertiaRatio()
			    //parameters.minInertiaRatio();//float 	minInertiaRatio()
			    
			    //parameters.minRepeatability(2);//long 	minRepeatability() ?
			  
			    */
			   
			   // -- initialisation de opencv_features2d.SimpleBlobDetector
			   opencv_features2d.SimpleBlobDetector detector= new opencv_features2d.SimpleBlobDetector(parameters); 

			   if (debug) {
			   //---------- affichages des paramètres utilisés ----------------- 
			   PApplet.println ("********** Paramètres utilisés par le SimpleBlobDetector **********"); 
			   PApplet.println ("maxThreshold = "+ parameters.maxThreshold());// float 	maxThreshold()
			   PApplet.println ("minThreshold = "+ parameters.minThreshold());// float 	minThreshold()
			   PApplet.println ("thresholdStep = "+ parameters.thresholdStep());// float 	thresholdStep()
			   PApplet.println(); 
			   
			   PApplet.println ("minDistBetweenBlobs = "+ parameters.minDistBetweenBlobs());//float 	minDistBetweenBlobs()
			   PApplet.println(); 
			    
			   PApplet.println ("filterByColor = "+ parameters.filterByColor());//boolean 	filterByColor()
			   PApplet.println ("blobColor = "+ (parameters.blobColor()&0xFF));//byte 	blobColor() - &0xFF pour affichage unsigned
			   PApplet.println(); 
			    
			   PApplet.println ("filterByArea = "+ parameters.filterByArea());//boolean 	filterByArea()
			   PApplet.println ("maxArea = "+ parameters.maxArea());//float 	maxArea()
			   PApplet.println ("minArea = "+ parameters.minArea());//float 	minArea()
			   PApplet.println(); 
			    
			   PApplet.println ("filterByCircularity = "+ parameters.filterByCircularity());//boolean 	filterByCircularity()
			   PApplet.println ("maxCircularity = "+ parameters.maxCircularity());//float 	maxCircularity()
			   PApplet.println ("minCircularity = "+ parameters.minCircularity());//float 	minCircularity()
			   PApplet.println(); 
			    
			   PApplet.println ("filterByConvexity = "+ parameters.filterByConvexity());//boolean 	filterByConvexity()
			   PApplet.println ("maxConvexity = "+ parameters.maxConvexity());//float 	maxConvexity()
			   PApplet.println ("minConvexity = "+ parameters.minConvexity());//float 	minConvexity()
			   PApplet.println(); 
			    
			   PApplet.println ("filterByInertia = "+ parameters.filterByInertia());//boolean 	filterByInertia()
			   PApplet.println ("maxInertiaRatio = "+ parameters.maxInertiaRatio());//float 	maxInertiaRatio()
			   PApplet.println ("minInertiaRatio = "+ parameters.minInertiaRatio());//float 	minInertiaRatio()
			   PApplet.println(); 
			    
			   PApplet.println ("minRepeatability = "+ parameters.minRepeatability());//long 	minRepeatability() ?
			    
			   } // fin if debug

			  //--- Etape 1 : Détecter les points-clé (Keypoints) en utilisant le SimpleBlob Detector

			  //opencv_features2d.KeyPoint keypoints_object = new opencv_features2d.KeyPoint(); 
			  opencv_features2d.KeyPoint keypoints_scene = new opencv_features2d.KeyPoint(); 
			  
			  // void 	detect(opencv_core.CvArr image, opencv_features2d.KeyPoint keypoints, opencv_core.CvArr mask) 
			  //detector.detect(iplImgObject, keypoints_object, null); 
			  detector.detect(iplImgSceneIn, keypoints_scene, null); 

			
			  //--- récupération information sur les points clés unitaires du vecteur Keypoint scene

			  if (debug) PApplet.println ("Taille vecteur de points-clés objet ="+ keypoints_scene.capacity()); 
			  
			  for (int i=0; i<keypoints_scene.capacity(); i++) { // défile les point-clés unitaires au sein du vecteur de Keypoint
			    
			    keypoints_scene.position(i); // se positionne sur le points-clé voulu 
			    
			    if (debug) PApplet.print("Point clé "+ i +" : x="+keypoints_scene.pt().x()+ " | y="+keypoints_scene.pt().y()); // coordonnées du point			    
			    if (debug) PApplet.print( " | taille="+keypoints_scene.size()); // taille 
			    if (debug) PApplet.print( " | angle="+keypoints_scene.angle()); // angle
			    if (debug) PApplet.println ( " | octave="+keypoints_scene.octave()); // octave
			  
			    
			   
			    // voir drawKeypoints
			    
			    //p.stroke(0,0,255);
			    //p.noFill();
			    //ellipse (keypoints_scene.pt().x(), keypoints_scene.pt().y(), 5,5); 
			    //p.ellipse (keypoints_scene.pt().x(), keypoints_scene.pt().y(), keypoints_scene.size(),keypoints_scene.size()); 

			    //noFill();
			    //ellipse (keypoints_object.pt().x(), keypoints_object.pt().y(), keypoints_object.size(),keypoints_object.size()); 
			    
			    // récupération des valeurs dans le tableau de Keypoints... 
			    
			    pointKeypoint.x=(int)keypoints_scene.pt().x(); 			  
			    pointKeypoint.y=(int)keypoints_scene.pt().y(); 
			    
			    sizeKeypoint=keypoints_scene.size();
			    angleKeypoint=keypoints_scene.angle();
			    octaveKeypoint=keypoints_scene.octave();
			    
				  //ajoute un keypoint à ArrayList de keypoint
				  myKeypointsListScene.add(new Keypoint(new Point (pointKeypoint.x, pointKeypoint.y), sizeKeypoint, angleKeypoint,octaveKeypoint));

			    
			    
			  } // fin for 
			  
			  
			  
			// ----------------- renvoi de la fonction ----------------- 

			//--- récupère le ArrayList dans un tableau 
			//Blob[] myBlobs = (Blob[]) selectBlobsList.toArray(new Blob[0]);// récupère le ArrayList dans le tableau de Blob 
			Keypoint[] myKeypoints = (Keypoint[]) myKeypointsListScene.toArray(new Keypoint[0]);// récupère le ArrayList dans le tableau de Blob
			
			  //-- debug - vérification du contenu du tableau obtenu par toArray - 
			PApplet.println("Nombre de Keypoints : " + myKeypoints.length); //-- affiche nombre de keypoints sélectionnés
			
			// renvoie le tableau de Keypoints
			return(myKeypoints); 
			
			
		} // fin fonction keypointsSBD 
		
		//--- fonc keypointsSBD minimale 
		public Keypoint[] keypointsSBD (
				
				opencv_core.IplImage iplImgSceneIn,  
				boolean debug) { // la fonction Blobs renvoie un tableau de blobs

			return(keypointsSBD (		
				iplImgSceneIn,  
				150, 250, 10, // float minThreshold, float maxThreshold,  float thresholdStep,
				1, //float minDistBetweenBlobs,
				true, 255, // boolean filterByColor, int blobColor,
				false, 0,0, // boolean filterByArea, float minArea, float maxArea, 
				false, 0,0,// boolean filterByCircularity, float minCircularity, float maxCircularity,
				false,0,0, // boolean filterByConvexity, float minConvexity, float maxConvexity,
				false, 0,0, // boolean filterByInertia, float minInertiaRatio, float maxInertiaRatio,
				2,// long minRepeatability,
				debug// boolean debug) 
				)
				); // fin return
		
		} // fin keypointsSBD
		
		/////////////// Détection de lignes avec Hough Standard //////////////
		
		//-------- fonction detectLines principale -------- 
		public Line[] detectLines ( // détection de ligne avec algorithme de Hough version Standard
				opencv_core.IplImage iplImgIn, // image source 
				// int method, // la méthode à utiliser - Standard, ou probabilistic - ici obligatoirement Standard - probabilistic dans detectVertices
				double rhoIn, // la résolution pour la longueur du vecteur normal
				double thetaIn, // la résolution angulaire reçue en degré - convertit en radians pour la fonction native OpenCV
				float thresholdCannyIn, // ajout : 1er seuil du filtre Canny utilisé. Le second seuil Canny vaut seuilCannyIn/2 - Canny pas utilisé ici si =0
				int thresholdAccumulatorIn, // seuil pour l'accumulateur - droite sélectionnée seulement si nombre vote > seuil 
				//double param1, // paramètre de méthode - =0 avec méthode standard, = longueur minimale segments avec probabilistic
				// double param2, // paramètre de méthode - =0 avec méthode standard, = écart min entre 2 segments avec probabilistic
				boolean debug // drapeau affichage messages
				) 
		{

		
			// détection des cercles dans une image en se basant sur la fonction native HoughCircles
			// qui se base sur la transformée de Hough
			
			// la fonction reçoit : 
			// opencv_core.IplImage iplImgIn : image source 

			// boolean debug : drapeau affichage messages
		
			//---- arrayList utilisé pour la détection 
			myLinesList = new ArrayList<Line>(); // alternative à un tableau fixe = créer un ArrayList de Keypoint  - déclaré ici en static - cf entête classe 
			
			//--- les éléments du Circle courant ---
		    //Point  point1Line=new Point(); // 
		    //Point  point2Line=new Point(); // 
			
			//------ chaque ligne est définie par son vecteur normal : 1 angle et 1 longueur 
			float thetaLine=0;
			float rhoLine=0;
			
		    //-------------- bascule l'image en niveau de gris ---------- 
		    gray(iplImgIn); // convertit l'image en niveau de gris ...
		    
		    //--- on passe par le buffer interne Trans8U1C pour la suite de la fonction 
		    opencv_core.cvCopy(BufferGray, Trans8U1C); // copie l'image Ipl en entrée dans le IplImage destination

		    
		    /* -- alternative sans passer par bufferGray ---
			opencv_imgproc.cvCvtColor(iplImgIn, Trans8U1C, opencv_imgproc.CV_RGB2GRAY); // bascule en niveaux de gris 

			opencv_imgproc.cvCvtColor(Trans8U1C, iplImgIn, opencv_imgproc.CV_GRAY2RGB); // rebascule en RGB 
			// les 3 canaux du buffer RGB sont identiques = l'image est en niveaux de gris 

			// la copie est conservée dans le buffer Trans8U1C

		    */
		    
		    //--------- +/- application filtre Canny sur l'image ---------- 
		    
		    if (thresholdCannyIn!=0) { // si thresholdCannyIn différent de zéro
		    	
		    	// on applique le filtre Canny avant détection de lignes
		    	canny( Trans8U1C, thresholdCannyIn, thresholdCannyIn/2,3); 
		    	
		    } // fin if 
		    
		    //-------------- détection des lignes dans l'image et création des objets Line correspondants ---------- 
		    
		    //------- détection des lignes avec méthodes standard
		    
		    opencv_core.CvMemStorage storage = opencv_core.CvMemStorage.create(); // crée le CvMemStorage utilisé par la fonction 
		    	
		  //--- récupère les paramètres du vecteur normal de chaque ligne détectées dans un CvSeq 
		    // static opencv_core.CvSeq 	cvHoughLines2(opencv_core.CvArr image, com.googlecode.javacpp.Pointer line_storage, int method, double rho, double theta, int threshold, double param1, double param2) 
		    opencv_core.CvSeq lines=opencv_imgproc.cvHoughLines2( 
		    		Trans8U1C, // image source 
		    		storage, // MemStorage de stockage du CvSeq renvoyé
		    		opencv_imgproc.CV_HOUGH_STANDARD, // méthode utilisée - standard renvoie rho et theta pour chaque ligne
		    		rhoIn , // résolution rho
		    		PApplet.radians((float)thetaIn), // résolution theta en radians 
		    		thresholdAccumulatorIn, // seuil accumulateur
		    		0, // paramètre méthode - =0 avec CV_HOUGH_STANDARD 
		    		0 // paramètre méthode = 0 avec CV_HOUGH_STABDARD
		    		);  // fin 

		    
		    if (debug) PApplet.println("Nombre de lignes détecté =" + lines.total()); 

		    for(int i = 0; i < lines.total(); i++ )
		    { 
		        opencv_core.CvPoint2D32f point = new opencv_core.CvPoint2D32f(opencv_core.cvGetSeqElem(lines, i)); // récupère l'élément du cvseq comme un point2D = 2 valeurs 32F 

		              rhoLine=point.x(); 
		              thetaLine=point.y(); 
		              
		              PApplet.println ("rho= " + rhoLine + " theta= " +thetaLine ); 

		              //---- ajoute l'objet Line à la ArrayList
		              myLinesList.add(new Line(thetaLine, rhoLine, 240)); // Line (float thetaIn, float rhoIn, float YMaxIn = heightImage)

		    } // fin for 
  
			// ----------------- renvoi de la fonction ----------------- 

			//--- récupère le ArrayList dans un tableau 
			//Blob[] myBlobs = (Blob[]) selectBlobsList.toArray(new Blob[0]);// récupère le ArrayList dans le tableau de Blob 
		    Line[] myLines; 
		  	myLines = (Line[]) myLinesList.toArray(new Line[0]);// récupère le ArrayList dans le tableau de Circle
			
			  //-- debug - vérification du contenu du tableau obtenu par toArray - 
			PApplet.println("Nombre de Lignes : " + myLines.length); //-- affiche nombre de Circles sélectionnés
			
			// renvoie le tableau de Circles
			return(myLines); 
			
			
		} // fin de la fonction detectLines
		
		//----------------- fonction drawLines : pour dessiner un tableau d'objets Lines --------------------- 
		
		public void drawLines( 
				Line[] linesIn, // le tableau d'objets Lines à tracer 
				int xRefIn, int yRefIn, // coordonnées référence objet
				float scaleIn, // facteur d'échelle à utilisr - 1 par défaut  
				int colorStrokeIn, int strokeWeightIn, // paramètres graphiques
				boolean debug // debug
				)
		{

			if (linesIn!=null) { // si le tableau de Line n'est pas vide 
				
			 // fixe les paramètres graphiques 
	          p.stroke(colorStrokeIn); // couleur verte
	          p.strokeWeight(strokeWeightIn); // épaisseur pourtour 

	          
	        for (int i=0; i<linesIn.length; i++) { // passe en revue les éléments du tableau 
	            

	            p.line(xRefIn+(linesIn[i].pointY0.x*scaleIn), yRefIn+(linesIn[i].pointY0.y*scaleIn), // point intersection avec droite y=0
	            		xRefIn+(linesIn[i].pointYMax.x*scaleIn), yRefIn+(linesIn[i].pointYMax.y*scaleIn) // point intersection avec droite y=YMax
	            		); // trace lignes 
	            
	          } // fin for
	        
		} // fin if linesIn!=null

		} // fin drawLines 
		
		//----------------- fonction drawLine : pour dessiner une ligne à partir de 2 points --------------------- 
		
		public void drawLine( 
				Point pointDebutIn, // le point de début de la ligne
				Point pointFinIn, // le point de fin de la ligne
				int xRefIn, int yRefIn, // coordonnées référence objet
				float scaleIn, // facteur d'échelle à utilisr - 1 par défaut  
				int colorStrokeIn, int strokeWeightIn, // paramètres graphiques
				boolean debug // debug
				)
		{

				
			 // fixe les paramètres graphiques 
	          p.stroke(colorStrokeIn); // couleur verte
	          p.strokeWeight(strokeWeightIn); // épaisseur pourtour 

	          	            
	          // trace la ligne 
	           p.line(xRefIn+(pointDebutIn.x*scaleIn), yRefIn+(pointDebutIn.y*scaleIn), // point intersection avec droite y=0
	            		xRefIn+(pointFinIn.x*scaleIn), yRefIn+(pointFinIn.y*scaleIn) // point intersection avec droite y=YMax
	            		); // trace ligne 
	            
	          
		} // fin drawLine 
			
		//----------------- fonction drawLine : pour dessiner un objet Line --------------------- 
		
		public void drawLine( 
				Line lineIn, // l'objets Line à tracer 
				int xRefIn, int yRefIn, // coordonnées référence objet
				float scaleIn, // facteur d'échelle à utilisr - 1 par défaut  
				int colorStrokeIn, int strokeWeightIn, // paramètres graphiques
				boolean debug // debug
				)
		{

				
			 // fixe les paramètres graphiques 
	          p.stroke(colorStrokeIn); // couleur verte
	          p.strokeWeight(strokeWeightIn); // épaisseur pourtour 

	          	            
	          // trace la ligne 
	           p.line(xRefIn+(lineIn.pointY0.x*scaleIn), yRefIn+(lineIn.pointY0.y*scaleIn), // point intersection avec droite y=0
	            		xRefIn+(lineIn.pointYMax.x*scaleIn), yRefIn+(lineIn.pointYMax.y*scaleIn) // point intersection avec droite y=YMax
	            		); // trace ligne 
	            
	          
		} // fin drawLine 
		
		
		/////////////// Détection de segments avec Hough Probabilistic //////////////
		
		
		/////////////// Détection de cercles avec Hough //////////////
		
		//-------- fonction detectCircles principale -------- 
		public Circle[] detectCircles (
				opencv_core.IplImage iplImgIn, // image source 
				float dpIn, // coeff diviseur de la résolution de l'accumulateur dans l'espace de Hough
				float minDistIn, // distance minmale entre 
				float thresholdCannyIn, //  1er seuil du filtre Canny utilisé. Le second seuil Canny vaut seuilCannyIn/2
				float thresholdAccumulatorIn, // seuil utilisé par l'accumulateur pour prise en compte des centres des cercles. 
				int minRadiusIn, // rayon minimum - mettre 0 par défaut
				int maxRadiusIn, // rayon maximum - mettre 0 par défaut
				boolean debug // drapeau affichage messages
				) 
		{
			
			// détection des cercles dans une image en se basant sur la fonction native HoughCircles
			// qui se base sur la transformée de Hough
			
			// la fonction reçoit : 
			// opencv_core.IplImage iplImgIn : image source 
			// float dpIn : coeff diviseur de la résolution de l'accumulateur dans l'espace de Hough
			// float minDistIn : distance minmale 
			// float thresholdCannyIn : 1er seuil du filtre Canny utilisé. Le second seuil Canny vaut seuilCannyIn/2
			// float seuilAccumulatorIn : seuil utilisé par l'accumulateur pour prise en compte des centres des cercles. 
			// float minRadiusIn : rayon minimum - mettre 0 par défaut
			// float maxRadiusIn : rayon maximum - mettre 0 par défaut
			// boolean debug : drapeau affichage messages
		
			//---- arrayList utilisé pour la détection 
			myCirclesList = new ArrayList<Circle>(); // alternative à un tableau fixe = créer un ArrayList de Keypoint  - déclaré ici en static - cf entête classe 
			
			//--- les éléments du Circle courant ---
		    Point  centerCircle=new Point(); // point du Keypoint
		    float radiusCircle; // rayon du point clé

		    //-------------- bascule l'image en niveau de gris ---------- 
		    gray(iplImgIn); // convertit l'image en niveau de gris ...
		    
		    //--- on passe par le buffer interne Trans8U1C pour la suite de la fonction 
		    opencv_core.cvCopy(BufferGray, Trans8U1C); // copie l'image Ipl en entrée dans le IplImage destination

		    
		    /* -- alternative sans passer par bufferGray ---
			opencv_imgproc.cvCvtColor(iplImgIn, Trans8U1C, opencv_imgproc.CV_RGB2GRAY); // bascule en niveaux de gris 

			opencv_imgproc.cvCvtColor(Trans8U1C, iplImgIn, opencv_imgproc.CV_GRAY2RGB); // rebascule en RGB 
			// les 3 canaux du buffer RGB sont identiques = l'image est en niveaux de gris 

			// la copie est conservée dans le buffer Trans8U1C

		    */
		    
		    
		    //-------------- détection des cercle dans l'image et création des objets Circle correspondants ---------- 
		    
		    //------- détection des cercles 
		    // source : http://code.google.com/p/javacv/wiki/ConvertingOpenCV
		    
		    opencv_core.CvMemStorage storage = opencv_core.CvMemStorage.create(); // crée le CvMemStorage utilisé par la fonction 
		    
		    
		    //--- récupère les cercles détectés dans un CvSeq
		    //static opencv_core.CvSeq 	cvHoughCircles(opencv_core.CvArr image, com.googlecode.javacpp.Pointer circle_storage, int method, double dp, double min_dist, double param1, double param2, int min_radius, int max_radius) 
		    opencv_core.CvSeq circles = opencv_imgproc.cvHoughCircles(
		    		Trans8U1C, storage, 
		    		opencv_imgproc.CV_HOUGH_GRADIENT, 
		    		(double) dpIn, 
		    		(double) minDistIn, 
		    		(double) thresholdCannyIn, 
		    		(double) thresholdAccumulatorIn, 
		    		 minRadiusIn,
		    		 maxRadiusIn );

		    if (debug) PApplet.println("Nombre de cercles détecté =" + circles.total()); 

		   //float[] xyr = new float[3] ; // pas nécessaire ici 
		   
		   
		      for(int i = 0; i < circles.total(); i++ ){ // on défile les éléments de la séquence = les cerles détectés
				    // chaque éléments de la séquence est un ensemble de 3 float à priori... x, y et r des cercles détectés

		    	  
		        opencv_core.CvPoint3D32f point = new opencv_core.CvPoint3D32f(opencv_core.cvGetSeqElem(circles, i)); // récupère l'élément du cvseq comme un point3D 
		        // = 3 valeurs 32F  = bien vu !! 
		        // solution trouvée ici : http://stackoverflow.com/questions/8198106/using-warpperspective-on-a-sequence-of-points-given-by-houghcircles-opencv 

	              centerCircle.x=(int)point.x(); 
		           centerCircle.y=(int)point.y(); 
		              
		              radiusCircle=point.z(); 
		              
		              if (debug) PApplet.println (" center: x= " + centerCircle.x + " | y= " +  centerCircle.y + " | radius = " + radiusCircle ); 
		              
		              if (debug) p.stroke(255,0,0);
		              if (debug) p.noFill(); 
		              if (debug) p.ellipse ( centerCircle.x, centerCircle.y, radiusCircle, radiusCircle); 
		              
	              //---- ajoute le Circle à la ArrayList
		              myCirclesList.add(new Circle(new Point (centerCircle.x, centerCircle.y), radiusCircle));
		              
		    } // fin for 
		   
		    
			// ----------------- renvoi de la fonction ----------------- 

			//--- récupère le ArrayList dans un tableau 
			//Blob[] myBlobs = (Blob[]) selectBlobsList.toArray(new Blob[0]);// récupère le ArrayList dans le tableau de Blob 
		    Circle[] myCircles; 
		  	myCircles = (Circle[]) myCirclesList.toArray(new Circle[0]);// récupère le ArrayList dans le tableau de Circle
			
			  //-- debug - vérification du contenu du tableau obtenu par toArray - 
			PApplet.println("Nombre de Circles : " + myCircles.length); //-- affiche nombre de Circles sélectionnés
			
			// renvoie le tableau de Circles
			return(myCircles); 
			
			
		} // fin de la fonction detectCircles
		
		//------ detectCircles : forme simplifiée ------------- 
		public Circle[] detectCircles (
				opencv_core.IplImage iplImgIn, // image source 
				boolean debug // drapeau affichage messages
				) 
		{
		
			return( detectCircles (
					iplImgIn, // opencv_core.IplImage iplImgIn, // image source 
					1, // float dpIn, // coeff diviseur de la résolution de l'accumulateur dans l'espace de Hough
					20, // float minDistIn, // distance minmale entre 
					200, // float thresholdCannyIn, //  1er seuil du filtre Canny utilisé. Le second seuil Canny vaut seuilCannyIn/2
					100, // float thresholdAccumulatorIn, // seuil utilisé par l'accumulateur pour prise en compte des centres des cercles. 
					0, // int minRadiusIn, // rayon minimum - mettre 0 par défaut
					0, // int maxRadiusIn, // rayon maximum - mettre 0 par défaut
					debug// boolean debug // drapeau affichage messages
					) 
					); // fin return
			
		} // fin detectCircles simplifiée 
		
		//------- fonction houghCircles - idem detectCircles mais avec meme nom que fonction native openCV 
		public Circle[] houghCircles (
				opencv_core.IplImage iplImgIn, // image source 
				float dpIn, // coeff diviseur de la résolution de l'accumulateur dans l'espace de Hough
				float minDistIn, // distance minmale entre 
				float thresholdCannyIn, //  1er seuil du filtre Canny utilisé. Le second seuil Canny vaut seuilCannyIn/2
				float thresholdAccumulatorIn, // seuil utilisé par l'accumulateur pour prise en compte des centres des cercles. 
				int minRadiusIn, // rayon minimum - mettre 0 par défaut
				int maxRadiusIn, // rayon maximum - mettre 0 par défaut
				boolean debug // drapeau affichage messages
				) 
		{
			return ( detectCircles ( // appelle detectCirles avec tous les paramètres idem... 
					iplImgIn,// opencv_core.IplImage iplImgIn, // image source 
					dpIn,// float dpIn, // coeff diviseur de la résolution de l'accumulateur dans l'espace de Hough
					minDistIn, // float minDistIn, // distance minmale entre 
					thresholdCannyIn, // float thresholdCannyIn, //  1er seuil du filtre Canny utilisé. Le second seuil Canny vaut seuilCannyIn/2
					thresholdAccumulatorIn, // float thresholdAccumulatorIn, // seuil utilisé par l'accumulateur pour prise en compte des centres des cercles. 
					minRadiusIn, // int minRadiusIn, // rayon minimum - mettre 0 par défaut
					maxRadiusIn,// int maxRadiusIn, // rayon maximum - mettre 0 par défaut
					debug// boolean debug // drapeau affichage messages
					)
					); // fin return
			
			
		}// fin houghCircles 
		
		/////////////// --- SURF Detector ----- ////////////////////
		
		//------- initialisation du SURF detector ---------- 
		public void initSURF(				
				float hessianThreshold, // entre 300 et 500 - 400 par défaut 
				int octaves, // 4 par défaut 
				int nOctaveLayers, // 2 par défaut 
				boolean upright // false si prise en compte de l'orientation, true sinon = plus rapide  
				) 
		{
			
			  //int minHessian=400; 
			  //detectorSURF=new opencv_features2d.SurfFeatureDetector();   
			  detectorSURF=new opencv_features2d.SurfFeatureDetector(hessianThreshold,octaves,nOctaveLayers,upright); 


		} // fin initialisation SURF Detector
		
		// -- initSURF - forme simple 
		public void initSURF() {
			
			initSURF (
					400, //float hessianThreshold, // entre 300 et 500 - 400 par défaut 
					4, //int octaves, // 4 par défaut 
					2, //int nOctaveLayers, // 2 par défaut 
					false // boolean upright // false si prise en compte de l'orientation, true sinon = plus rapide  
					);
			
		} // fin initSURF - forme réduite 
		
		//------- fonction principale keypoints SURF
		public Keypoint[] keypointsSURF ( // la fonction keypointsSURF renvoie un tableau d'objets Keypoint calculé par l'algorithme SURF
				opencv_core.IplImage iplImgIn,  // image au format natif opencv Iplimage
				int xRefIn, int yRefIn, float scaleIn, // paramètres pour tracer points natifs si debug true 
				boolean modeIn, // fixe le type de points clés - false = object - true = scene - identifiant internes possibles SCENE et OBJECT
				boolean debug, // si message debug
				boolean drawDebug // si dessin de debug
				) 
		{ 

			// OBJECT et SCENE sont 2 boolean déclarés en début de classe OpenCV
			if (modeIn==OBJECT) myKeypointsListObject = new ArrayList<Keypoint>(); // alternative à un tableau fixe = créer un ArrayList de Keypoint  - déclaré ici en static - cf entête classe 
			if (modeIn==SCENE) myKeypointsListScene = new ArrayList<Keypoint>(); // alternative à un tableau fixe = créer un ArrayList de Keypoint  - déclaré ici en static - cf entête classe 

			//--- les éléments du keypoint courant ---
		    Point  pointKeypoint=new Point(); // point du Keypoint
		    float sizeKeypoint; // rayon du point clé
		    float angleKeypoint; // angle du point clé
		    int octaveKeypoint; //octave du point clé

		    //--- Etape 1 : Détecter les points-clé (Keypoints) en utilisant le SURF Detector
		    
		    // ---> voir initSURF()
			  //int minHessian=400; 
			  //opencv_features2d.SurfFeatureDetector detectorSURF=new opencv_features2d.SurfFeatureDetector();   
			  //opencv_features2d.SurfFeatureDetector detector=new opencv_features2d.SurfFeatureDetector(500,2,4,false); 

			  opencv_features2d.KeyPoint keypointsOut = new opencv_features2d.KeyPoint(); 
			  
			  // void 	detect(opencv_core.CvArr image, opencv_features2d.KeyPoint keypoints, opencv_core.CvArr mask) 
			  detectorSURF.detect(iplImgIn, keypointsOut, null); 

			  //--- récupération information sur les points clés unitaires du vecteur Keypoint objet

				  
				 if(debug && modeIn==OBJECT) PApplet.println ("Taille vecteur de points-clés  objet ="+ keypointsOut.capacity()); 
				 if(debug && modeIn==SCENE) PApplet.println ("Taille vecteur de points-clés  scene ="+ keypointsOut.capacity()); 
			  
			  for (int i=0; i<keypointsOut.capacity(); i++) { // défile les point-clés unitaires au sein du vecteur de Keypoint
			    
			    keypointsOut.position(i); // se positionne sur le points-clé voulu 
			    
			    if(debug)PApplet.print("Point clé "+ i +" : x="+keypointsOut.pt().x()+ " | y="+keypointsOut.pt().y()); // coordonnées du point
			    if(debug)PApplet.print( " | taille="+keypointsOut.size()); // taille 
			    if(debug)PApplet.print( " | angle="+keypointsOut.angle()); // angle
			    if(debug)PApplet.println ( " | octave="+keypointsOut.octave()); // octave

			    //--- tracé des points clés natifs si drawDebug--- 
			    if(drawDebug) p.stroke(255,0,0);
			    if(drawDebug) p.ellipse (xRefIn+(keypointsOut.pt().x()*scaleIn), yRefIn+(keypointsOut.pt().y()*scaleIn), 5,5); 
			    
			    //-------- récupère le Keypoint dans le ArrayList -------- 
			    pointKeypoint.x=(int)keypointsOut.pt().x(); 			  
			    pointKeypoint.y=(int)keypointsOut.pt().y(); 
			    
			    sizeKeypoint=keypointsOut.size();
			    angleKeypoint=keypointsOut.angle();
			    octaveKeypoint=keypointsOut.octave();
			    
				  //ajoute un keypoint à ArrayList de keypoint
			    if (modeIn==OBJECT) myKeypointsListObject.add(new Keypoint(new Point (pointKeypoint.x, pointKeypoint.y), sizeKeypoint, angleKeypoint,octaveKeypoint));
			    if (modeIn==SCENE) myKeypointsListScene.add(new Keypoint(new Point (pointKeypoint.x, pointKeypoint.y), sizeKeypoint, angleKeypoint,octaveKeypoint));

			  } // fin for  
			  
			    
			  //----- mémorise les keypoints natifs pour utilisation ultérieure 
			  if (modeIn==OBJECT) keypointsObject=keypointsOut; // mémorise les points clés objets
			  if (modeIn==SCENE) keypointsScene=keypointsOut; // mémorise les points clés objets
			
			  //-- debug interne 

			  //if(debug) PApplet.println ("Taille vecteur de points-clés objet ="+ keypointsObject.capacity()); 			  
			  //int i=3;
			  //keypointsObject.position(i); // se positionne sur le points-clé voulu 
			  //if(debug)PApplet.print("Point clé "+ i +" : x="+keypointsObject.pt().x()+ " | y="+keypointsObject.pt().y()); // coordonnées du point
			
				// ----------------- renvoi de la fonction ----------------- 

				//--- récupère le ArrayList dans un tableau 
				//Blob[] myBlobs = (Blob[]) selectBlobsList.toArray(new Blob[0]);// récupère le ArrayList dans le tableau de Blob 
			  Keypoint[] myKeypoints; 
			  	if (modeIn==OBJECT) myKeypoints = (Keypoint[]) myKeypointsListObject.toArray(new Keypoint[0]);// récupère le ArrayList dans le tableau de Blob
			  	else myKeypoints = (Keypoint[]) myKeypointsListScene.toArray(new Keypoint[0]);// récupère le ArrayList dans le tableau de Blob
				
				  //-- debug - vérification du contenu du tableau obtenu par toArray - 
				PApplet.println("Nombre de Keypoints : " + myKeypoints.length); //-- affiche nombre de keypoints sélectionnés
				
				// renvoie le tableau de Keypoints
				return(myKeypoints); 
				

			  
		} // fin keypointsSURF
		
		
		//------- fonction simple keypointsSURF = keypoints Speed Up Robust Features
		public Keypoint[] keypointsSURF (
				opencv_core.IplImage iplImgIn,  // image au format natif opencv Iplimage
				boolean modeIn, // fixe le type de points clés - false = object - true = scene
				boolean debug // drapeau de message des debug
				) 
		{ 
		
			return(keypointsSURF (
					iplImgIn,  // image au format natif opencv Iplimage
					0,0,1,// int xRefIn, int yRefIn, float scaleIn, // paramètres pour tracer points natifs si debug true 
					modeIn,// fixe le type de points clés - false = object - true = scene
					debug, //boolean debug
					false //boolean drawDebug
					)//fin keypointSURF
					);// fin return
			
		
		} // fin KeyPoint SURF 
		
		//---------- fonction matchSURF() = recherche des correspondances entre les points clés --------------------
		//public void detectMatchSURF(opencv_core.IplImage iplImgObject, int xRefObject, int yRefObject, opencv_core.IplImage iplImgScene, int xRefScene, int yRefScene, boolean debug) { 
		
		public void detectMatchSURF(
				opencv_core.IplImage iplImgObject, // image objet au format natif opencv iplimage
				int xRefObject, int yRefObject, // coordonnées de référence de l'image objet - utilisées si drawDebug=true
				opencv_core.IplImage iplImgScene, // image scene au format natif opencv iplimage
				int xRefScene, int yRefScene, // coordonnées du coin sup gauche de l'image scene - utilisées si drawDebug=true
				boolean debug, // drapeau aiichage des messages
				boolean drawDebug // drapeau affichage dessin avec valeur native opencv
				) {
			
			// cette fonction reçoit les 2 images objet et scene
			// les coordonnées coin sup gauche de la scene
			// drapeau de debug
			
			// cette fonction se base sur les vecteurs de points clés interne keypointsScene et keypointsObjet
			// qui auront été initialisés par la fonction keypointsSURF() qui aura été appelée avant 
			
			// cette fonction détecte les correspondances existantes 
			
			// les points sont mis dans les CvMat keypoints initialisés avec keypointsSURF
			
			
		
			  // Etape 2 : Calcul des descripteurs des 2 images 
			  
			  opencv_features2d.SurfDescriptorExtractor extractor=new opencv_features2d.SurfDescriptorExtractor(); // crée un objet Extracteur de descripteur
			  
			  opencv_core.CvMat descriptors_object= new opencv_core.CvMat(null); // crée objet CvMat descripteur objet // Use null pointer... thanks Samuel ! 
			  //opencv_core.CvMat descriptors_object = null; // crée objet CvMat descripteur objet - ne marche pas !!
			  
			  opencv_core.CvMat descriptors_scene = new opencv_core.CvMat(null) ; // crée objet CvMat descripteur objet
			  
			  // --- Important : réinitialiser la position des vecteur de points-clés 
			  keypointsObject.position(0); // se repositionne au début
			  keypointsScene.position(0); // se repositionne au début
			  
			  //  void 	compute(opencv_core.CvArr image, opencv_features2d.KeyPoint keypoints, opencv_core.CvMat descriptors)  // forme 1 image 
			  extractor.compute( iplImgObject, keypointsObject, descriptors_object ); // calcule le descripteur 
			  extractor.compute( iplImgScene, keypointsScene, descriptors_scene ); // calcule le descripteur 

			      
			      //--- affichageinfo des CvMat des descripteurs
			      
			      if (debug) {
			    	  PApplet.println ("--- Descripteur objet --- "); 
			      
			    	  PApplet.println ("Taille descripteur objet ="+ descriptors_object.rows()); // chaque ligne du descripteur correspond à 1 descripteur d'un point clé
			    	  PApplet.println ("Nombre canaux ="+ descriptors_object.channels()); 
			    	  PApplet.println ("Taille Element="+ descriptors_object.elemSize()); 
			    	  PApplet.println ("Nombre colonnes="+ descriptors_object.cols()); // le descripteur a 64 éléments ou 128 selon paramétrage de extended de SURF 
			    	  PApplet.println(); 

			    	  PApplet.println ("--- Descripteur scene --- "); 
			    	  PApplet.println ("Taille descripteur scene ="+ descriptors_scene.rows()); 
			      
			      } // fin if debug
			      

			   // Etape 3 : Mise en correspondance (Match) des descripteurs 
			   
			   opencv_features2d.FloatL2BruteForceMatcher matcher = new opencv_features2d.FloatL2BruteForceMatcher(); 
			   
			   //opencv_features2d.DMatch matches = new opencv_features2d.DMatch(); 
			   matches = new opencv_features2d.DMatch(); 

			   //void 	match(opencv_core.CvArr queryDescriptors, opencv_core.CvArr trainDescriptors, opencv_features2d.DMatch matches, opencv_core.CvArr mask)    
			   matcher.match(descriptors_object, descriptors_scene, matches, null);
			   
			 // affichage des DMatch unitaire du vecteur de DMatch obtenu 
			
			   //comptGoodMatch=0; // initialise le nombre de GoodMatch 
			   
			   //---- initialise le vecteur de Keypoint natif pour le stockage des keypoint Match Object
			   //keypointsObjectMatch=new opencv_features2d.KeyPoint(); 
			   //keypointsObjectMatch.capacity(matches.capacity()); // crée un tableau de keypoints de la taille du vecteur de DMatch
			   
			   //---- initialise le vecteur de Keypoint natif pour le stockage des keypoint Match Scene			   
			   //keypointsSceneMatch=new opencv_features2d.KeyPoint(); // marche pas
			   //keypointsSceneMatch.capacity(matches.capacity()); // crée un tableau de keypoints de la taille du vecteur de DMatch // marce pas 
			   
			   for( int i = 0; i < matches.capacity(); i++ ) { // défile le vecteur de DMatch 
				   
				    //opencv_features2d.DMatch myDMatch = new opencv_features2d.DMatch(); // non 
				    //myDMatch=matches.at(0); // non - 
				    
				    matches.position(i); // se positionne sur le DMatch voulu dans le vecteur de DMatch appelé matches

				    float distf=matches.distance();
				
				    
				    //--- infos pour debug --- 
				    if (debug)PApplet.print("Matches " + i + " : distance =" + distf); // distance du Dmatch courant
				    if (debug)PApplet.print(" | index objet =" + matches.queryIdx() ); // index du descripteur scene
				    if (debug)PApplet.println(" | index scene =" + matches.trainIdx() ); // index du descripteur objet
				    
				    
				    // le code qui suit assure la transposition du keypoint(queryIdx) vers keypointMatch(i).. etc..
				    
				    //--- obtention coordonnées point clé objet
				    keypointsObject.position(matches.queryIdx()); // se positionne sur le points-clé voulu du keypointObject
				    if (debug)PApplet.println("Point clé objet "+ matches.queryIdx() +" : x="+keypointsObject.pt().x()+ " | y="+keypointsObject.pt().y()); // coordonnées du point    


/*									    
				    //--- stockage dans le vecteur de keypointObjectMatch -- marche pas 
				    keypointsObjectMatch.position(i); // se positionne à la position i 
				    keypointsObjectMatch.pt().x(keypointsObject.pt().x()); // met la valeur du keypoints queryIdx dans keypoints match(i)
				    keypointsObjectMatch.pt().y(keypointsObject.pt().y()); // met la valeur du keypoints queryIdx dans keypoints match(i)
				    keypointsObjectMatch.octave(keypointsObject.octave()); // met la valeur du keypoints queryIdx dans keypoints match(i)
				    keypointsObjectMatch.angle(keypointsObject.angle()); // met la valeur du keypoints queryIdx dans keypoints match(i)
				    keypointsObjectMatch.size(keypointsObject.size()); // met la valeur du keypoints queryIdx dans keypoints match(i)

				    if (debug)PApplet.println("Point clé objet "+i +" : x="+keypointsObjectMatch.pt().x()+ " | y="+keypointsObjectMatch.pt().y()); // coordonnées du point    
*/ 

				    //--- obtention coordonnées point clé scene
				    keypointsScene.position(matches.trainIdx()); // se positionne sur le points-clé voulu 
				    if (debug) PApplet.println("Point clé scene "+ matches.trainIdx() +" : x="+keypointsScene.pt().x()+ " | y="+keypointsScene.pt().y()); // coordonnées du point    

				  //--- stockage dans le vecteur de keypointSceneMach - marche pas 
				    //keypointsSceneMatch.position(i); // se positionne à la position i 
				    //keypointsSceneMatch.pt().x(keypointsScene.pt().x()); // met la valeur du keypoints queryIdx dans keypoints match(i)
				    
				    
				    //----------- sélection des points de correspondance ---------- 
				    // if (matches.distance()<distanceMaxGoodMatch) { // critère de sélection

				      // affiche les DMatch retenus et la ligne de correspondance
				    	 if (drawDebug)p.stroke(0,0,255);
				    	 if (drawDebug) p.fill(0,0,255); 
				    	 if (drawDebug) p.ellipse (xRefObject+keypointsObject.pt().x(), yRefObject+keypointsObject.pt().y(), 5,5); // point objet
				    	 if (drawDebug) p.ellipse (xRefScene+keypointsScene.pt().x(), yRefScene+keypointsScene.pt().y(), 5,5); // point objet
				    	 if (drawDebug) p.line (xRefObject+keypointsObject.pt().x(), yRefObject+keypointsObject.pt().y(), xRefScene+keypointsScene.pt().x(), yRefScene+keypointsScene.pt().y()); // ligne entre les 2 points

				      //comptGoodMatch=comptGoodMatch+1; // incrémente variable de comptage des DMatch retenus
				      
				    // } // fin if distance
				        
				    
				  }  //fin for défile le vecteur de DMatch

			   /// if (debug)PApplet.println("Nombre de concordances utiles ( Good Match ) =" +  comptGoodMatch);

			   //opencv_features2d.drawMatches(iplImgObject, keypointsObject, iplImgScene, keypointsScene, matches, iplImgDest, opencv_core.cvScalar(0,0,255,0), opencv_core.cvScalar(255,0,0,0), null,1); 

/*			   
			   //------------ renvoi de la fonction -------- 
			   Point[][] pointsOut = new Point[2][matches.capacity()];
			   
			   for (int i=0; i<matches.capacity(); i++) { // défile les éléments du CvMat matches
				   
				   pointsOut[0][i]=new Point((int)keypointsObject.pt().x(), (int)keypointsObject.pt().y()); // met les points clé objet dans [0][i]
				   pointsOut[1][i]=new Point((int)keypointsScene.pt().x(), (int)keypointsScene.pt().y()); // met les points clé scène dans [0][i]
				  				   
			   }
			   	
			   return (pointsOut); 
*/			           
			   

		} // fin detectMatchSURF 
		
		//-------- detectMatchSURF simplifiée ------ 
		public void detectMatchSURF(
				opencv_core.IplImage iplImgObjectIn, // image objet au format natif opencv iplimage
				opencv_core.IplImage iplImgSceneIn, // image scene au format natif opencv iplimage
				boolean debugIn // drapeau affichage des messages
				) {
			
			detectMatchSURF( // detectMatchSURF uniquement à parir Images et boolean debug
					iplImgObjectIn, // opencv_core.IplImage iplImgObject, // image objet au format natif opencv iplimage
					0,0, // int xRefObject, int yRefObject, // coordonnées de référence de l'image objet - utilisées si drawDebug=true
					iplImgSceneIn,// opencv_core.IplImage iplImgScene, // image scene au format natif opencv iplimage
					0,0, // int xRefScene, int yRefScene, // coordonnées du coin sup gauche de l'image scene - utilisées si drawDebug=true
					debugIn, // boolean debug, // drapeau aiichage des messages
					false// boolean drawDebug // drapeau affichage dessin avec valeur native opencv
					);

		} // fin detectMatchSURF simplifié

		
		//--------- fonction extraction des points clés objet des Match SURF --------------
		// -- la fonction detectMatch devra avoir été appelée juste avant --

		public Keypoint[] keypointsMatchSURF(
				boolean modeIn, // fixe le type de points clés - false = object - true = scene
				boolean debug // drapeau d'affichage des messages - 
				) {
			
			   
			   Keypoint[] keypointsOut = new Keypoint[matches.capacity()];

			   if (modeIn==OBJECT) {
				   
				   if(debug)PApplet.println("========== Les points-clés de concordance (Match) de l'objet ============="); 
				   
				   for (int i=0; i<matches.capacity(); i++) { // défile les éléments du CvMat matches
				   
					   matches.position(i); // défile les DMacth du vecteur de DMatch 
					   
					   keypointsObject.position(matches.queryIdx()); // se positionne sur le bon Keypoint du vecteur de Keypoints
				   
					   //--- création du Keypoint du tableau en se basant sur le keypoint natif du vecteur de keypoints 
					   keypointsOut[i]=new Keypoint(new Point ((int)keypointsObject.pt().x(), (int)keypointsObject.pt().y()),keypointsObject.size(), keypointsObject.angle(), keypointsObject.octave() ); // met les points clé objet dans keypoints[i]
				   
					   //--- si debug -- 
					   if(debug)PApplet.print("Point clé "+ i +" : x="+keypointsOut[i].point.x + " | y="+keypointsOut[i].point.y); // coordonnées du point
					   if(debug)PApplet.print( " | taille="+keypointsOut[i].size); // taille 
					   if(debug)PApplet.print( " | angle="+keypointsOut[i].angle); // angle
					   if(debug)PApplet.println ( " | octave="+keypointsOut[i]); // octave
				  
				  				   
				   }// fin for
			   
			   } // fin if OBJECT
			   	

			   else  { // si modeIn== SCENE

				   if(debug)PApplet.println("========== Les points-clés de concordance (Match) de la scène ============="); 

				   for (int i=0; i<matches.capacity(); i++) { // défile les éléments du CvMat matches

					   matches.position(i); // défile les DMacth du vecteur de DMatch 

					   keypointsScene.position(matches.trainIdx()); // se positionne sur le bon Keypoint du vecteur de Keypoints
				   
					   keypointsOut[i]=new Keypoint(new Point ((int)keypointsScene.pt().x(), (int)keypointsScene.pt().y()),keypointsScene.size(), keypointsScene.angle(), keypointsObject.octave() ); // met les points clé objet dans keypoints[i]
				   
					   //--- si debug -- 
					   if(debug)PApplet.print("Point clé "+ i +" : x="+keypointsOut[i].point.x + " | y="+keypointsOut[i].point.y); // coordonnées du point
					   if(debug)PApplet.print( " | taille="+keypointsOut[i].size); // taille 
					   if(debug)PApplet.print( " | angle="+keypointsOut[i].angle); // angle
					   if(debug)PApplet.println ( " | octave="+keypointsOut[i]); // octave
				  				  
				  				   
				   }// fin for
			   
			   } // fin if SCENE
			   
			   //--- renvoi de la fonction 
			   return (keypointsOut); 
	
		} // fin keypointsObjectMatchSURF
		
		
		//--------- drawLinesMatchSURF : trace les lignes de concordance entre les points clés ------------ 
		public boolean drawLinesMatchSURF (
				Keypoint[] keypointsObjectMatchIn, // le tableau de Keypoint match de l'objet 
				int xRefObject, int yRefObject, // coordonnées référence objet
				Keypoint[] keypointsSceneMatchIn, // le tableau de Keypoint match de la scene
				int xRefScene, int yRefScene, // coordonnées référence scene
				float scaleIn, int colorStrokeIn, int strokeWeightIn, // paramètres graphiques
				boolean debug // debug
				)
		{
			
			// la fonction reçoit les tableaux de Keypoints Match de la scene et de l'objet 
			// ---------- la fonction va tracer les concordances --------
			
			//--- test préalable --- 
			if (keypointsObjectMatchIn.length!=keypointsSceneMatchIn.length){ // si les tableaux de Keypoint n'ont pas la meme taille 
				if (debug) PApplet.println("Les tableaux de points-clé ne sont pas de la même taille : tracé des concordances impossible !");
				return(false); // sort de la fonction en renvoyant false
			}
			else { // si les tableaux de Keypoint ont bine la même taille 
				
				if (debug) PApplet.println("Les tableaux de points-clé sont de la même taille : tracé des concordances possible !");
				
				 // fixe les paramètres graphiques 
		          p.stroke(colorStrokeIn); // couleur verte
		          p.strokeWeight(strokeWeightIn); // épaisseur pourtour 

				for (int i=0; i<keypointsSceneMatchIn.length; i++) { // on défile les keypoints 
					
					//---- trace la ligne entre 2 points de meme index des 2 tableaux 
					p.line(xRefObject+(keypointsObjectMatchIn[i].point.x*scaleIn), yRefObject+(keypointsObjectMatchIn[i].point.y*scaleIn), 
							xRefScene+(keypointsSceneMatchIn[i].point.x*scaleIn), yRefScene+(keypointsSceneMatchIn[i].point.y*scaleIn));
					
				}
				
				return (true); // sortie en renvoyant true 
			}
			
		}
		
		//---------- selectGoodMatchSURF() ------------------- 
		public void selectGoodMatchSURF (
				float distanceMax, // valeur de la "distance" à utiliser pour la sélection - entre 0 et 1
				int xRefObject, int yRefObject, // coordonnées de référence de l'image objet - utilisé si drawDebug=true
				int xRefScene, int yRefScene,  // coordonnées de référence de l'image objet - utilisé si drawDebug=true
				boolean debug, // drapeau d'affichage des messages
				boolean drawDebug // drapeau d'affichage du dessin d'après valeur natives
				) 
		{
			
			// cette fonction compte les correspondances utiles et sélectionne les points clés objet et scène correspondants
			
			// les points sont mis dans les CvMat keypoints initialisés avec keypointsSURF
			// Les CvMat keypoints ne contiennent alors que les valeurs des keypoints sélectionnés
			
			  distanceMaxGoodMatch= distanceMax; // Initialise distance de sélection des gooMatch 
			
			   comptGoodMatch=0; // initialise le nombre de GoodMatch 

			   //========== Etape 4 : Sélection et comptage des GoodMatch = les correspondances utiles  en se basant sur distance Max ==================

			   for( int i = 0; i < matches.capacity(); i++ ) { // défile le vecteur de DMatch 
				   
				    //opencv_features2d.DMatch myDMatch = new opencv_features2d.DMatch(); // non 
				    //myDMatch=matches.at(0); // non - 
				    
				    matches.position(i); // se positionne sur le DMatch voulu dans le vecteur de DMatch appelé matches

				    float distf=matches.distance();
				
				    
				    //--- infos pour debug --- 
				    if (debug)PApplet.print("Matches " + i + " : distance =" + distf); // distance du Dmatch courant
				    if (debug)PApplet.print(" | index objet =" + matches.queryIdx() ); // index du descripteur scene
				    if (debug)PApplet.println(" | index scene =" + matches.trainIdx() ); // index du descripteur objet
				    
				    //--- obtention coordonnées point clé objet
				    keypointsObject.position(matches.queryIdx()); // se positionne sur le points-clé voulu 
				    if (debug)PApplet.println("Point clé objet "+ matches.queryIdx() +" : x="+keypointsObject.pt().x()+ " | y="+keypointsObject.pt().y()); // coordonnées du point    

				    //--- obtention coordonnées point clé scene
				    keypointsScene.position(matches.trainIdx()); // se positionne sur le points-clé voulu 
				    if (debug) PApplet.println("Point clé scene "+ matches.trainIdx() +" : x="+keypointsScene.pt().x()+ " | y="+keypointsScene.pt().y()); // coordonnées du point    

				   
				    
				    //----------- sélection des points de correspondance ---------- 
				    if (matches.distance()<distanceMaxGoodMatch) { // critère de sélection

				      // affiche les DMatch retenus et la ligne de correspondance
				    	 //if (debug)p.stroke(0,255,0);
				    	 //if (debug) p.fill(0,255,0); 
				    	 //if (debug) p.ellipse (xRefObject+keypointsObject.pt().x(), yRefObject+keypointsObject.pt().y(), 5,5); // point objet
				    	 //if (debug) p.ellipse (xRefScene+keypointsScene.pt().x(), yRefScene+keypointsScene.pt().y(), 5,5); // point objet
				    	 //if (debug) p.line (xRefObject+keypointsObject.pt().x(), yRefObject+keypointsObject.pt().y(), xRefScene+keypointsScene.pt().x(), yRefScene+keypointsScene.pt().y()); // ligne entre les 2 points

				      comptGoodMatch=comptGoodMatch+1; // incrémente variable de comptage des DMatch retenus
				      
				    } // fin if distance
				        
				    
				  }  //fin for défile le vecteur de DMatch
			   
			   if (debug)PApplet.println("Nombre de concordances utiles ( Good Match ) =" +  comptGoodMatch);
			   	

			 //========== Récupération des coordonnées des points Objet et Scène des GoodMatch (correspondances utiles sélectionnées) ==================
	
			   if (comptGoodMatch<1) {
				   
				   PApplet.println("Aucune correspondance utile trouvée (GoodMatch) détecté !");
			   }
			   else { // si au moins 1 GoodMatch 
				   

		        //--- crée 2 CvMat de stockage des points 32F - 2 canaux pour utilisation avec FindHomography
		        // FindHomography reçoit 2 CvMat de n points de 1 ligne et n colones et 2 canaux en 32F.. 1 canal pour les x et 1 canal pour les y 
		      
		      // static opencv_core.CvMat 	cvCreateMat(int rows, int cols, int type) 
		      //opencv_core.CvMat cvMatPointsObject= opencv_core.cvCreateMat(comptGoodMatch,1,opencv_core.CV_32FC2); // crée un cvMat de point - pas bonne structure
		      //opencv_core.CvMat cvMatPointsScene= opencv_core.cvCreateMat(comptGoodMatch,1,opencv_core.CV_32FC2); // crée un cvMat de point - pas bonne structure
		      
		      // static opencv_core.CvMat 	create(int rows, int cols, int depth, int channels) // forme la plus claire
		      //opencv_core.CvMat cvMatGoodMatchObject= opencv_core.CvMat.create (1,comptGoodMatch,opencv_core.CV_32F, 2); 
			   cvMatGoodMatchObject= opencv_core.CvMat.create (1,comptGoodMatch,opencv_core.CV_32F, 2);
			   
		      // opencv_core.CvMat cvMatGoodMatchScene= opencv_core.CvMat.create (1,comptGoodMatch,opencv_core.CV_32F, 2);
			   cvMatGoodMatchScene= opencv_core.CvMat.create (1,comptGoodMatch,opencv_core.CV_32F, 2);
		      
		      if (debug) PApplet.print("CvMat Points Objet : canaux=" + cvMatGoodMatchObject.channels()); 
		      if (debug) PApplet.println(" | lignes =" + cvMatGoodMatchObject.rows()); 

		      if (debug) PApplet.print("CvMat Points Scene : canaux=" + cvMatGoodMatchScene.channels()); 
		      if (debug) PApplet.println(" | lignes =" + cvMatGoodMatchScene.rows()); 


		   //---------- on re-passe en revue les Dmatch pour ne retenir que certains : 2ème passage : met points dans les CvMat -------

		      if (debug) PApplet.println(); 
		      if (debug) PApplet.println("------------ Liste des DMatch sélectionnés -----------"); 
		   
		   comptGoodMatch=0; // RAZ compte good Match
		   
		  for( int i = 0; i < matches.capacity(); i++ ) { // défile le vecteur de DMatch 
		  
		    matches.position(i); // se positionne sur le DMatch voulu dans le vecteur de DMatch appelé matches

		    if (matches.distance()<distanceMaxGoodMatch) { // critère de sélection - arbitraire ici 

		    	if (debug) PApplet.println("Point sélectionné "+ comptGoodMatch +" : "); 

		        float distf=matches.distance();
		        
		        //--- infos pour debug --- 
		        if (debug) PApplet.print("Matches " + i + " : distance =" + distf); // distance du Dmatch courant
		        if (debug) PApplet.print(" | index objet =" + matches.queryIdx() ); // index du descripteur scene
		        if (debug) PApplet.println(" | index scene =" + matches.trainIdx() ); // index du descripteur objet
		        
		        //--- obtention coordonnées point clé objet
		        keypointsObject.position(matches.queryIdx()); // se positionne sur le points-clé voulu 
		        if (debug) PApplet.println("Point clé objet "+ matches.queryIdx() +" : x="+keypointsObject.pt().x()+ " | y="+keypointsObject.pt().y()); // coordonnées du point    
		    
		        //--- obtention coordonnées point clé scene
		        keypointsScene.position(matches.trainIdx()); // se positionne sur le points-clé voulu 
		        if (debug) PApplet.println("Point clé scene "+ matches.trainIdx() +" : x="+keypointsScene.pt().x()+ " | y="+keypointsScene.pt().y()); // coordonnées du point    
		    
		        // -- stocke les points dans les CvMat de points --- 

		        //cvMatPointsScene.put(comptGoodMatch, 0 ,0, (double)keypoints_scene.pt().x()); // met la valeur de x dans le premier canal - pas bon 

		/*
		        //-- 1ère option - qui marche .. 
		        //--- cvMat objet         
		        double[] xyPointObject = {keypoints_object.pt().x(),keypoints_object.pt().y()}; 
		        cvMatPointsObject.put(comptGoodMatch,xyPointObject); // met la valeur double  x,y dans le CvMat
		        
		        xyPointObject[0]=0; xyPointObject[1]=0; // RAZ tableau
		        cvMatPointsObject.get(comptGoodMatch,xyPointObject); // récupère valeur depuis le cvMat
		        println(" CvMat Object" + comptGoodMatch +" : x= "+xyPointObject[0]+" | y= "+xyPointObject[1]); 

		        //--- CvMat scene ---
		        double[] xyPointScene = {keypoints_scene.pt().x(),keypoints_scene.pt().y()}; 
		        cvMatPointsScene.put(comptGoodMatch,xyPointScene); // met la valeur double  x,y dans le CvMat
		        
		        xyPointScene[0]=0; xyPointScene[1]=0; // RAZ tableau
		        cvMatPointsScene.get(comptGoodMatch,xyPointScene); // récupère valeur depuis le cvMat
		        println(" CvMat Scene " + comptGoodMatch +" : x= "+xyPointScene[0]+" | y= "+xyPointScene[1]); 
		*/

		        //-- 2ème option - plus simple une fois qu'on a compris la structure du CvMat

		        int index=comptGoodMatch*2; // calcul index = 2n
		        
		        //--- cvMat object         
		        cvMatGoodMatchObject.put(index,keypointsObject.pt().x()); // met la valeur x dans le CvMat - pointeur 2n (1er canal = les x )
		        cvMatGoodMatchObject.put(index+1,keypointsObject.pt().y()); // met la valeur x dans le CvMat - pointeur 2n + 1 ( 2ème canal = les y )
		                
		        if (debug) PApplet.println(" CvMat Object " + comptGoodMatch +" : x= "+cvMatGoodMatchObject.get(index) +" | y= "+cvMatGoodMatchObject.get(index+1)); 

		        //--- cvMat Scene        
		        cvMatGoodMatchScene.put(index,keypointsScene.pt().x()); // met la valeur x dans le CvMat - pointeur 2n (1er canal = les x )
		        cvMatGoodMatchScene.put(index+1,keypointsScene.pt().y()); // met la valeur x dans le CvMat - pointeur 2n + 1 ( 2ème canal = les y )
		                
		        if (debug) PApplet.println(" CvMat Object " + comptGoodMatch +" : x= "+cvMatGoodMatchScene.get(index) +" | y= "+cvMatGoodMatchScene.get(index+1)); 

		        // affiche les DMatch retenus et la ligne de correspondance
		        if (drawDebug) p.stroke(0,255,0);
		        if (drawDebug) p.fill(0,255,0); 
		        if (drawDebug) p.ellipse (xRefObject+keypointsObject.pt().x(), yRefObject+keypointsObject.pt().y(), 5,5); // point objet
		        if (drawDebug) p.ellipse (xRefScene+keypointsScene.pt().x(), yRefScene+keypointsScene.pt().y(), 5,5); // point objet
		        if (drawDebug) p.line (xRefObject+keypointsObject.pt().x(), yRefObject+keypointsObject.pt().y(), xRefScene+keypointsScene.pt().x(), yRefScene + keypointsScene.pt().y()); // ligne entre les 2 points
		    
		          comptGoodMatch=comptGoodMatch+1; // incrémente variable de comptage des DMatch retenus
		      
		    } // fin if distance
		        
		    
		  }  //fin for défile le vecteur de DMatch

		 } // fin else GoodMatch 

		} // fin selectGoodMatch
		

		//------- selectGoodMatchSURF() - forme minimale 

		public void selectGoodMatchSURF (
				float distanceMax, // valeur de la "distance" à utiliser pour la sélection - entre 0 et 1
				boolean debug // drapeau d'affichage des messages
				) 
		{
			
			selectGoodMatchSURF (
					distanceMax,//float distanceMax, // valeur de la "distance" à utiliser pour la sélection - entre 0 et 1
					0,0,//int xRefObject, int yRefObject, // coordonnées de référence de l'image objet - utilisé si drawDbug=true
					0,0,// int xRefScene, int yRefScene,  // coordonnées de référence de l'image objet - utilisé si drawDbug=true
					debug, //boolean debug, // drapeau d'affichage des messages
					false // drapeau d'affichage du dessin d'après valeur natives
					); 
			

		}

		
		//--------- fonction extraction des points clés objet des Good Match SURF --------------
		// -- la fonction selectGoodMatch devra avoir été appelée juste avant --
		
		public Keypoint[] keypointsGoodMatchSURF(
				boolean modeIn, // fixe le type de points clés - false = object - true = scene
				boolean debug // drapeau d'affichage des messages
				) 
		{
			
				// renvoie le tableau des Keypoints GoodMatch
				// doit être appelé après selectGoodMatch 
			   
			   Keypoint[] keypointsOut = new Keypoint[comptGoodMatch];
			   
			   int compt=0; // variable de sous-comptage des Dmatch utiles

			   if (modeIn==OBJECT) {			   
				   
				   //---- messages commun pour tous les points 
				   if(debug)PApplet.println("========== Les points-clés de concordance utiles (GoodMatch) de l'objet ============="); 
				   if (debug) PApplet.println("Critère de sélection : distance < "+ distanceMaxGoodMatch);
	
					   for( int i = 0; i < matches.capacity(); i++ ) { // défile le vecteur de DMatch 
						   
						    matches.position(i); // se positionne sur le DMatch voulu dans le vecteur de DMatch appelé matches

						    //----------- sélection des points de correspondance ---------- 
						    if (matches.distance()<distanceMaxGoodMatch) { // critère de sélection

								   keypointsObject.position(matches.queryIdx()); // se positionne sur le bon Keypoint du vecteur de Keypoints
								   
								   keypointsOut[compt]=new Keypoint(new Point ((int)keypointsObject.pt().x(), (int)keypointsObject.pt().y()),keypointsObject.size(), keypointsObject.angle(), keypointsObject.octave() ); // met les points clé objet dans keypoints[i]
							   
								   //--- si debug -- Attention : l'index ici n'est pas i mais compt !
								   if(debug)PApplet.print("Point clé "+ i +" : x="+keypointsOut[compt].point.x + " | y="+keypointsOut[compt].point.y); // coordonnées du point
								   if(debug)PApplet.print( " | taille="+keypointsOut[compt].size); // taille 
								   if(debug)PApplet.print( " | angle="+keypointsOut[compt].angle); // angle
								   if(debug)PApplet.println ( " | octave="+keypointsOut[compt]); // octave
							  
								   compt=compt+1; 
						      
						    } // fin if distance
						        
						    
						  }  //fin for défile le vecteur de DMatch
					   
					   if (debug)PApplet.println("Nombre de concordances utiles ( Good Match ) pour l'objet  =" +  comptGoodMatch);

			   
			   } // fin if OBJECT
			   	

			   else  { // si modeIn== SCENE
				   
				   //---- messages commun pour tous les points 
				   if(debug)PApplet.println("========== Les points-clés de concordance utiles (GoodMatch) de la scène ============="); 
				   if (debug) PApplet.println("Critère de sélection : distance < "+ distanceMaxGoodMatch);
	
				   	
					   for( int i = 0; i < matches.capacity(); i++ ) { // défile le vecteur de DMatch 
						   
						    matches.position(i); // se positionne sur le DMatch voulu dans le vecteur de DMatch appelé matches

						    //----------- sélection des points de correspondance ---------- 
						    if (matches.distance()<distanceMaxGoodMatch) { // critère de sélection

								   keypointsScene.position(matches.trainIdx()); // se positionne sur le bon Keypoint du vecteur de Keypoints
								   
								   keypointsOut[compt]=new Keypoint(new Point ((int)keypointsScene.pt().x(), (int)keypointsScene.pt().y()),keypointsScene.size(), keypointsScene.angle(), keypointsScene.octave() ); // met les points clé objet dans keypoints[i]
							   
								   //--- si debug --  Attention : l'index ici n'est pas i mais compt !
								   if(debug)PApplet.print("Point clé "+ i +" : x="+keypointsOut[compt].point.x + " | y="+keypointsOut[compt].point.y); // coordonnées du point
								   if(debug)PApplet.print( " | taille="+keypointsOut[compt].size); // taille 
								   if(debug)PApplet.print( " | angle="+keypointsOut[compt].angle); // angle
								   if(debug)PApplet.println ( " | octave="+keypointsOut[compt]); // octave
							  
								   compt=compt+1; 
						      
						    } // fin if distance
						        
						    
						  }  //fin for défile le vecteur de DMatch
					   
					   if (debug)PApplet.println("Nombre de concordances utiles ( Good Match ) pour la scene =" +  comptGoodMatch);
		   
			   } // fin if SCENE
			   
			   
			   
			   //--- renvoi de la fonction 
			   return (keypointsOut); 
	
		} // fin keypointsObjectMatchSURF
		
		//------- fonction detectObjectIntoSceneSURF') --------------
		
		public Point[] detectObjectIntoSceneSURF (
				opencv_core.IplImage iplImgObjectIn, // image Objet de départ
				int xRefScene, int yRefScene, // coordonnées de l'image de la scène - utilisé si drawDebug=true
				boolean debug, // drapeau d'affichage des messages
				boolean drawDebug // drapeau d'affichage des dessin à partir données natives 
				) 
		{
			
			// cette fonction se base sur les CvMat interne de points "Good Match" objet et scene 
			// pour déterminer dans un premier temps la transformation de perspective entre l'objet et l'objet dans la scene
			// la transformation de perspective obtenue est un matrice 3x3 (un CvMat)
			
			// connaissant la transformation de perspective, on recalcule alors les coordonnées du cadre entourant l'objet dans la scene.
			
			Point[] pointsOut=new Point[4]; // crée un tableau de 4 éléments 
			
			   if (comptGoodMatch<4) {
				   
				   PApplet.println("Pas assez correspondance utiles (GoodMatch) détectées (4 minimum) !");
				   
				   //--------- renvoi de la fonction -------------- 
				   
				   //Point[] pointsOut=new Point[4]; // crée un tableau de 4 éléments 
				   
				   //--- rempli le tableau de 0 si pas assez de GoodMatch
				   
				   for (int n=0; n<4; n++) { // défile les points 
					   if (debug) PApplet.println ("Point " + n + " :  x= "+ 0 +" | y= " + 0 ); // affiche coordonnées du point 


					   pointsOut[n]=new Point(0,0);
					   
					   //pointsOut[n].setLocation(myX, myX); // pas bon... 
					   //pointsOut[n].x=(int)cornersScene.get(2*n); // pas bon...
					   //pointsOut[n].y=(int)cornersScene.get((2*n)+1); // pas bon... 
					   			     
				   } // fin for

			   }
			   else { // si au moins 1 GoodMatch 

			   //========== extraction de la transformation de perspective entre l'objet et la scene ================ 

			  // une fois que l'on dispose des 2 CvMat de points, utiliser la fonction findHomography pour retrouver la transformation de perspective entre l'objet et l'objet dans la scene
			  // static int 	cvFindHomography(opencv_core.CvMat src_points, opencv_core.CvMat dst_points, opencv_core.CvMat homography) 
			  // static int 	cvFindHomography(opencv_core.CvMat src_points, opencv_core.CvMat dst_points, opencv_core.CvMat homography, int method, double ransacReprojThreshold, opencv_core.CvMat mask) 
			  // cf http://opencv.itseez.com/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#findhomography 
			  
			  
			     opencv_core.CvMat H3x3= opencv_core.CvMat.create(3,3); // crée objet CvMat pour la matrice de la transformation Homographique
			     
			     if (debug) PApplet.println ("CvMatObject Size : "+(cvMatGoodMatchObject.size()/8) + "| Channels : " + cvMatGoodMatchObject.channels()); 
			     if (debug) PApplet.println ("CvMatScene Size : "+(cvMatGoodMatchScene.size()/8) + "| Channels : " + cvMatGoodMatchScene.channels()); 
			     
			     //opencv_calib3d.cvFindHomography( cvMatPointsObject, cvMatPointsScene, cvMatHomoGraph); 
			     //opencv_calib3d.cvFindHomography(cvMatPointsObject, cvMatPointsScene, H3x3, 0, 1.0, null);  // Tous les points
			     opencv_calib3d.cvFindHomography(cvMatGoodMatchObject, cvMatGoodMatchScene, H3x3, opencv_calib3d.CV_RANSAC, 1.0, null);  // RANSAC - bon résultat 
			     //opencv_calib3d.cvFindHomography(cvMatPointsObject, cvMatPointsScene, H3x3, opencv_calib3d.CV_LMEDS, 1.0, null);  // LMEDS
			     
			   // la fonction findHomography renvoie une matrice de transformation des points de l'objet en points de la scene
			   // cette matrice pourra être utilisée avec la fonction perspectiveTransform (module core - opérations sur tableaux)
			   // pour recalculer les points de son choix de l'objet dans la scene
			   // classiquement le cadre entourant l'objet pourra être recalculé dans la scène
			   // http://opencv.itseez.com/modules/core/doc/operations_on_arrays.html#perspectivetransform


			    //============ calcul du cadre entourant l'objet en se basant sur la transformation de perspective ========== 

			    // défintion des coins de l'image objet dans un CvMat     

			    // création d'un CvMat de stockage des points objet : 4 groupes de 2 valeurs = 1 lignes x 4 colonnes x 2 canaux = 8 pointeurs
			    opencv_core.CvMat cornersObject= opencv_core.CvMat.create(1,4,opencv_core.CV_32F,2); // crée un CvMat de 4 points - 32F - 2 canaux


			    //--- coin sup gauche --- 
			    double[] xyPoint = {0,0}; // tableau de 2 valeurs x,y du point    
			    cornersObject.put(0,xyPoint) ; // met les 2 valeurs à l'index 2n et 2n+1
			    //cornersObject.put(0,iplImgSrc.height()) ; // met la valeur à l'index 2n

			    // alternative basée sur CvPoint2D32f - en fait moins "propre"
			    //opencv_core.CvPoint2D32f myPoint = new opencv_core.CvPoint2D32f(0,0); // déclare un vecteur de CvPoint2D32F - 1 seul élément
			    //cornersObject.put(0,myPoint.x()) ; // met les 2 valeurs à l'index 2n et 2n+1
			    //cornersObject.put(1,myPoint.y()) ; 
			     
			    //--- coin sup droit 
			    xyPoint=new double[] {iplImgObjectIn.width(),0}; // tableau de 2 valeurs x,y du point
			    //xyPoint[0]=iplImgObject.width(); xyPoint[1]=0; // tableau de 2 valeurs x,y du point
			    cornersObject.put(2,xyPoint) ; // met les 2 valeurs à l'index 2n et 2n+1
			   
			    //--coin inf droit 
			    xyPoint=new double[] {iplImgObjectIn.width(),iplImgObjectIn.height()}; // tableau de 2 valeurs x,y du point
			    //xyPoint[0]=iplImgObject.width(); xyPoint[1]=iplImgObject.height(); // tableau de 2 valeurs x,y du point
			    cornersObject.put(4,xyPoint) ; // met les 2 valeurs à l'index 2n et 2n+1

			    //--coin inf gauche 
			    xyPoint=new double[] {0,iplImgObjectIn.height()}; // tableau de 2 valeurs x,y du point
			    //xyPoint[0]=0; xyPoint[1]=iplImgObject.height(); // tableau de 2 valeurs x,y du point
			    cornersObject.put(6,xyPoint) ; // met les 2 valeurs à l'index 2n et 2n+1


			   //--- affiche le contenu du CVMat --- 
			   for (int n=0; n<4; n++) { // défile les points du cadre de l'image de départ 
				   if (debug) PApplet.println ("Point " + n + " :  x= "+ cornersObject.get(2*n)+" | y= " + cornersObject.get((2*n)+1)); 
				   if (drawDebug) p.point((float)cornersObject.get(2*n),(float)cornersObject.get((2*n)+1));
			     
			   }

			    // création d'un CvMat de stockage des points Cadre dans la Scene : 4 groupes de 2 valeurs = 1 lignes x 4 colonnes x 2 canaux = 8 pointeurs
			    opencv_core.CvMat cornersScene= opencv_core.CvMat.create(1,4,opencv_core.CV_32F,2); // crée un CvMat de 4 points - 32F - 2 canaux


			    //--- calcul des points destination --- 
			    //static void 	cvPerspectiveTransform(opencv_core.CvArr src, opencv_core.CvArr dst, opencv_core.CvMat mat) 
			    opencv_core.cvPerspectiveTransform(cornersObject, cornersScene, H3x3); 


			   //--- affiche le contenu du CVMat destination et les points correspondants --- 
			   if (drawDebug) {
			   p.noFill(); 
			   p.strokeWeight(2); 
			   p.stroke(255,0,255);
			   
			   p.beginShape();

			   for (int n=0; n<4; n++) { // défile les points 
				   if (debug) PApplet.println ("Point " + n + " :  x= "+ cornersScene.get(2*n)+" | y= " + cornersScene.get((2*n)+1)); // x en indice 2n et y en indice 2n+1 du CvMat
			     p.point(xRefScene+(float)cornersScene.get(2*n),yRefScene+(float)cornersScene.get((2*n)+1));
			     p.vertex(xRefScene +(float)cornersScene.get(2*n), yRefScene+(float)cornersScene.get((2*n)+1));
			     
			   } // fin for
			   
			   p.endShape(PApplet.CLOSE);
			   
			   } // fin if drawDebug
	
			   
			   //--------- renvoi de la fonction -------------- 
			   
			   //Point[] pointsOut=new Point[4]; // crée un tableau de 4 éléments 
			   
			   
			   for (int n=0; n<4; n++) { // défile les points 
				   if (debug) PApplet.println ("Point " + n + " :  x= "+ cornersScene.get(2*n)+" | y= " + cornersScene.get((2*n)+1)); // affiche coordonnées du point 


				   pointsOut[n]=new Point((int)cornersScene.get(2*n),(int)cornersScene.get((2*n)+1));
				   
				   //pointsOut[n].setLocation(myX, myX); // pas bon... 
				   //pointsOut[n].x=(int)cornersScene.get(2*n); // pas bon...
				   //pointsOut[n].y=(int)cornersScene.get((2*n)+1); // pas bon... 
				   			     
			   } // fin for
			   
			   } // fin else = si au moins 1 GoodMatch 	   
			   
			   return(pointsOut); // renvoie un tableau de 4 points correspondant aux 4 points du cadre de l'objet dans la scene

			   
		} // fin detectObjectIntoSceneSURF
		
		//--- fonction detectObjectIntoSceneSURF() - forme simplifiée 
		
		public Point[] detectObjectIntoSceneSURF (
				opencv_core.IplImage iplImgObjectIn, // image Objet de départ
				boolean debug // drapeau d'affichage des messages
				) 
		{
			
			return ( detectObjectIntoSceneSURF ( // detectObjectIntoScene sans tracé natif 
					iplImgObjectIn,//opencv_core.IplImage iplImgObjectIn, // image Objet de départ
					0,0,//int xRefScene, int yRefScene, // coordonnées de l'image de la scène - utilisé si drawDebug=true
					debug, // boolean debug, // drapeau d'affichage des messages
					false//boolean drawDebug // drapeau d'affichage des dessin à partir données natives 
					) 
					);
			
		} // fin fonction detectObjectIntoSceneSURF() - forme simplifiée 
		
		//------------ fonction targetObjectIntoSceneSURF() : détection du centre des keypoints significatif de la scene 
		public Point targetObjectIntoSceneSURF( 
				Keypoint[] keypointsGoodMatchScene, // tableau de keypoints de concordances significative de la Scene
				int lengthMin, // nombre de keypoint minimum 
				boolean drawCircle, // drapeau dessin cercle target
				int xRefIn, int yRefIn, // les coordonnées du cercle à tracer - utilisé si drawCircle=true
				float scaleIn, // l'échelle à utiliser - utilisé si drawCircle=true
				int radius, // rayon à utiliser - utilisé si drawCircle=true
				int colorStrokeIn, int strokeWeightIn, // couleur et épaisseur du pourtour du cercle - utilisé si drawCircle=true
				boolean fillIn, int colorFillIn, // drapeau de remplissage et couleur de remplissage - utilisé si drawCircle=true
				boolean debug // drapeau d'affichage des messages
				) 
		{
			
			Point pointOut= new Point(); 
			
		    Point[] pointsGoodMatchScene=keypointsToPoints(keypointsGoodMatchScene); // récupère les points à partir des Keypoints
			
			if (keypointsGoodMatchScene.length>(lengthMin-1)) { // si au moins 5 points de GoodMatch
		          
				// -- renvoie le point "moyenne" du groupe de point 
		          pointOut=meanPoints(pointsGoodMatchScene,true); 
		          


		          //---- dessine cercle de cible 
		          //p.ellipse(xRefIn+(pointOut.x*scaleIn), yRefIn+(pointOut.y*scaleIn), radius,radius); 

		          /*
		          //---------- fixe les paramètres graphiques  -----------
		          if (fillIn) p.fill(colorFillIn) ; else p.noFill(); 
		          p.stroke(colorStrokeIn); // couleur verte
		          p.strokeWeight(strokeWeightIn); // épaisseur pourtour 
		          
			        //--- affiche infos sur le cercle-- 
		 		 if (debug) PApplet.println ("Trace cercle de centre : x="+ pointOut.x+" | y="+ pointOut.y); 
		 		 
		          //---- dessine cercle 
		          p.ellipse(xRefIn+(pointOut.x*scaleIn), yRefIn+(pointOut.y*scaleIn), radius,radius); 

*/
		          drawCircle (
		  				pointOut,//Point center, // le centre du cercle à tracer
		  				xRefIn, yRefIn, // int xRefIn, int yRefIn, // les coordonnées du cercle à tracer
		  				scaleIn, // float scaleIn, // l'échelle à utiliser
		  				radius, // int radius, // rayon à utiliser 
		  				colorStrokeIn, strokeWeightIn,//int colorStrokeIn, int strokeWeightIn, // couleur et épaisseur du pourtour du cercle
		  				fillIn, colorFillIn,// boolean fillIn, int colorFillIn, // drapeau de remplissage et couleur de remplissage
		  				debug // boolean debugIn // drapeau d'affichage des messages 
		  				);

		          
		        } // fin fi 
			
			return (pointOut); 
			
		}
		
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////// FONCTIONS PROCESSING UTILES AVEC ARTOOLKIT et vision réelle 3D ///////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//------- fonction de conversion de la taille en pixel en valeur de l'angle apparent ----- 
		// la fonction reçoit une taille en pixel correspondant à la taille d'un objet tel qu'il est vu par la webcam
		// la fonction renvoie la valeur de l'angle apparent correspondant
		
		// important : on suppose ici que la taille en pixel est centrée 
		// càd soit au centre image webcam ou soit parallèle à 0y et coupée par 0x en son milieu... 
		
		// en pratique, utiliser la hauteur plutôt que la largeur dans le cas des Markers 
		// en effet la largeur varie fortement en fonction de l'orientation du Marker
		// mais la hauteur ne varie peu (autrement que par la distance) si le marker est dans le plan vertical et l'axe de la camera orthogonale au plan vertical
		
		// pour les markers, utiliser même préférentiellement la hauteur médiane passant par le centre du marker
		// car en cas d'orientation très différente du plan de la webcam, on obtiendra une hauteur majorée/minorée sur les bords
		// la distance à calculer étant deplus la distance du centre du marker, utiliser la hauteur médiane. 

		public float pixelsToAngle(float pixelSizeIn, float ouvertureIn, float widthCaptureIn, boolean debugIn) {
		  // on suppose pixelSizeIn centré d'où pixelSizeIn/2 correspond à 1/2 angle apparent
		  // -- renvoie angle apparent en degrés --
		 
		  // on par projection : 
		  // X/2 =sin(radians(alpha/2))*widthCapture/2/sin(radians(ouverture/2)); // calcul projection de l'angle sur X  (calcul idem sur Y car webcam sphérique)
		  // d'où : 
		  // sin (radians(alpha/2) = X/2 * sin(radians(ouverture/2)) / widthCapture/2
		  // radians (alpha/2) = arcsin [ X/2 * sin(radians(ouverture/2)) / widthCapture/2 ] 
		  // et angle = 2 * angle/2 ! 
		  
		  float angleOut=0; 
		  
		  angleOut= (pixelSizeIn/2)/ (widthCaptureIn/2) * PApplet.sin(PApplet.radians(ouvertureIn/2)); //   
		  angleOut= PApplet.asin(angleOut); 
		  angleOut=PApplet.degrees(angleOut) *2; 
		  
		  if (debugIn) PApplet.println("pixels = " + pixelSizeIn + " => angle= " + angleOut + " degrés"); 
		  
		  return (angleOut); 
		  
		} // fin pixelsToAngle
		
		//------- calcul de la distance réelle à partir de l'angle apparent
		public float realDistance( float angleAppDegIn, float sizeMarkerIn, boolean debugIn) {
		  // la fonction reçoit l'angle apparent centré (soit sur 0,0) soit sur Ox  soit Oy et la taille connue du Marker 
		  // la fonction renvoie la distance du marker
		 
		  float distanceOut=(float)0.0; 

		  distanceOut= sizeMarkerIn/ (2* PApplet.tan(PApplet.radians(angleAppDegIn))); 

		  if (debugIn) PApplet.println ("angle apparent marker =" + angleAppDegIn + " | taille marker =" + sizeMarkerIn +"=> distance calculée = " + distanceOut ); 

		  return (distanceOut);

		} // fin realDistance
		
		
		//-------- affiche des Axes angulaires gradués X et Y sur l'image webcam (1 graduation = 1 degré ------- 
		// reçoit ouverture en degres de la largeur de l'image webcam
		
		// L'angle d'ouverture est calculable empririquement avec tan angle= largeur réelle / 2 * distance camera
		// exemple Logitech C270 : 1/2 largeur réelle = 83cm  distance = 200 cm
		// d'où tan angle = 0.415 et d'où angle = 22.53 deg

		public void drawAngularAxis(int widthCaptureIn, int heightCaptureIn, float ouvertureWidthIn,int xRefIn, int yRefIn, float scaleIn, int strokeXIn, int strokeYIn, int strokeWeightIn) { 
		  
		   p.strokeWeight(strokeWeightIn); // largeur trait
		    
		    p.stroke (strokeYIn); 
		    p.line (xRefIn+(widthCaptureIn/2*scaleIn), yRefIn+(0*scaleIn), 
		    		xRefIn+(widthCaptureIn/2*scaleIn),yRefIn+(heightCaptureIn*scaleIn)); // vertical Y
		    
		    p.stroke(strokeXIn); 
		    p.line (xRefIn+0, yRefIn+(heightCaptureIn/2*scaleIn), 
		    		xRefIn+(widthCaptureIn*scaleIn), yRefIn+(heightCaptureIn/2*scaleIn)); // horizontal X 
		   
		   // --- quadrillage des angles --- 
		   for (int i=0; i<ouvertureWidthIn/2; i++) {
		    
		     float angleX=PApplet.sin(PApplet.radians(i))*widthCaptureIn/2/PApplet.sin(PApplet.radians(ouvertureWidthIn/2)); // calcul x 

		     //---- tracé sur axe des X ---      
		     
		     p.stroke (strokeXIn); 
		     
		     // affiche entre 0 et + ouverture/2 deg
		     p.line (xRefIn+(((widthCaptureIn/2)+angleX)*scaleIn), yRefIn+(((heightCaptureIn/2)-5)*scaleIn),
		    		 xRefIn+(((widthCaptureIn/2)+angleX)*scaleIn), yRefIn+(((heightCaptureIn/2)+5)*scaleIn)); // dessine un petit trait vertical

		     // affiche entre 0 et - ouverture/2 deg
		     p.line (xRefIn+(((widthCaptureIn/2)-angleX)*scaleIn), yRefIn+(((heightCaptureIn/2)-5)*scaleIn),
		    		 xRefIn+(((widthCaptureIn/2)-angleX)*scaleIn), yRefIn+(((heightCaptureIn/2)+5)*scaleIn)); // dessine un petit trait vertical
		     
		     //--- tracé sur Axe des Y --- 
		     //-- on se base sur angleX car à priori la webcam est sphérique donc idem Y et X 
		     
		     p.stroke (strokeYIn); 

		     // affiche entre 0 et + ouverture/2 deg
		     p.line (xRefIn+(((widthCaptureIn/2)-5)*scaleIn), yRefIn+(((heightCaptureIn/2)+angleX)*scaleIn),
		    		 xRefIn+(((widthCaptureIn/2)+5)*scaleIn), yRefIn+(((heightCaptureIn/2)+angleX)*scaleIn)); // dessine un petit trait vertical

		     // affiche entre 0 et - ouverture/2 deg
		     p.line (xRefIn+(((widthCaptureIn/2)-5)*scaleIn), yRefIn+(((heightCaptureIn/2)-angleX)*scaleIn),
		    		 xRefIn+(((widthCaptureIn/2)+5)*scaleIn), yRefIn+(((heightCaptureIn/2)-angleX)*scaleIn)); // dessine un petit trait horizontal
		     
		   } // fin for 
		 
		} // fin drawAngularAxis
		
		//---- fonction affiche le système courant 3D -----

		public void drawCurrentSyst3D (int sizeIn,  int strokeXIn, int strokeYIn, int strokeZIn, int strokeWeightIn) {
		  
		      p.strokeWeight(strokeWeightIn); // largeur trait

		  
		       //--- affichage du repère 3D 0x, 0y, 0z courant = ici le repère de base 
		      p.stroke(strokeXIn); 
		      p.line (0,0,0, sizeIn,0,0); // axe des x
		      
		      p.stroke(strokeYIn); 
		      p.line (0,0,0, 0,sizeIn,0); // axe des y
		      
		      p.stroke(strokeZIn); 
		      p.line (0,0,0, 0,0,sizeIn); // axe des z
		 
		} // drawCurrentSyst3D
		
		
		//-------- affiche dans la console les valeurs de la matrice 3D 
		public void printPMatrix3D(PMatrix3D syst3DIn ) {
			
	     //--- affichage de la matrice de transformation du repère  ----
	     // la matrice de transformation du système 3D est une matrice 4x4 de la forme
	     //  __                  __
	     // |                      |
	     // |  m01  m02  m03  m04  |
	     // |  m01  m02  m03  m04  |
	     // |  m01  m02  m03  m04  |
	     // |  m01  m02  m03  m04  |
	     // |__                  __|
	     //
	     
	     PApplet.println ("--- Matrice 4x4 du système 3D  ---"); 
	     PApplet.println ( "|\t" +  syst3DIn.m00 + "\t" +  syst3DIn.m01 + "\t" +  syst3DIn.m02 + "\t" +  syst3DIn.m03 + "\t" +  " |"); // affiche valeur matrice - voir PMatrix3D
	     PApplet.println ( "|\t" +  syst3DIn.m10 + "\t" +  syst3DIn.m11 + "\t" +  syst3DIn.m12 + "\t" +  syst3DIn.m13 + "\t" +  " |"); // affiche valeur matrice - voir PMatrix3D
	     PApplet.println ( "|\t" +  syst3DIn.m20 + "\t" +  syst3DIn.m21 + "\t" +  syst3DIn.m22 + "\t" +  syst3DIn.m23 + "\t" +  " |"); // affiche valeur matrice - voir PMatrix3D
	     PApplet.println ( "|\t" +  syst3DIn.m30 + "\t" +  syst3DIn.m31 + "\t" +  syst3DIn.m32 + "\t" +  syst3DIn.m33 + "\t" +  " |"); // affiche valeur matrice - voir PMatrix3D
	 
		} // fin printMatrix3D 

		//-------- affiche dans la console les valeurs de la matrice 3D 
		public void printCurrentPMatrix3D() {
			
			PMatrix3D syst3DOut =null; // crée objet local PMAtrix3D
			
			syst3DOut=p.getMatrix (syst3DOut); // récupère la matrice courante
			
			printPMatrix3D( syst3DOut); // affiche la matrice courante
			
		}
		
		//---------------------- fonction de mise à jour des paramètres des markers nya détectés ---------- 
		public void updateMarkers(MultiMarker nyaIn, Marker[] markersIn, boolean debugIn) {
			//-- reçoit l'objet Multimarker utilisé pour la détection et le tableau d'objet Marker associé
			// -- chaque objet Marker comporte les champs de description associés au Marker 

			int numMarkers=markersIn.length; // mémorise le nombre marker
			

		    // défile les markers en mémoire dans le Multimarker
		    for (int i=0; i<numMarkers; i++) { // passe en revue les markers de référence 

		      // si le marker n'esxiste pas = n'a pas été détecté, ne rien faire = passer à l'index suivant
		      if ((!nyaIn.isExistMarker(i))) { continue; } // passe au marker suivant si le marker(i) n'est pas détecté
		      
		      

		     // ce qui suit n'est exécuté que si le marker existe 
		       
		      if (debugIn) PApplet.println ("Le marker " + markersIn[i].name + " est détecté."); 

		      if (debugIn) PApplet. println("seuil de confiance = " + nyaIn.getConfidence(i)); // affiche le seuil de confiance de détection 
		       
		       // get the four marker coordinates into an array of 2D PVectors
		        PVector[] pos2d = nyaIn.getMarkerVertex2D(i);// récupère les 4 coins dans un tableau de PVector. Coordonnées 2D - origine = 0,0 de l'image = coin sup gauche 
		        // les index des vecteurs renvoyés sont 0:coin sup gauche, 1:coin sup droit, 2:coin inf droit, 3:coin inf gauche 
		        
		        // draw each vector both textually and with a red dot
		        for (int j=0; j<pos2d.length; j++) {

		/*          
		          String s = "(" + int(pos2d[j].x) + "," + int(pos2d[j].y) + ")";
		          fill(255);
		          rect(pos2d[j].x, pos2d[j].y, textWidth(s) + 3, textAscent() + textDescent() + 3);
		          fill(0);
		          text(s, pos2d[j].x + 2, pos2d[j].y + 2);
		          fill(0, 0, 255);
		 */         
		          //p.stroke(255,255,0); 
		          //p.point(pos2d[j].x, pos2d[j].y);
		          //ellipse(pos2d[j].x, pos2d[j].y, 10, 10);
		          
		        	 if (debugIn) PApplet.println ("Coin " + j + " : x="+ pos2d[j].x + " | y=" + pos2d[j].y ); 
		          
		          // mémorise les valeurs dans les champs de l'objet Marker
		          markersIn[i].corners2D[j].x=(int) pos2d[j].x; 
		          markersIn[i].corners2D[j].y=(int) pos2d[j].y; 


		        } // fin for pos2d

		        //----- calcul hauteur et largeur du Marker2D --- 
		        //  float widthMarker= abs(pos2d[0].x - pos2d[1].x) ; // prend bord sup comme référence largeur
		        //  float heightMarker= abs(pos2d[1].y - pos2d[2].y) ; // prend bord droit comme référence hauteur

		        // -- calculs des autres points utiles du Marker --- 
		        //--- calcul du milieu du bord sup du marker 2D --- 
		        float xUpCenter= (pos2d[0].x + pos2d[1].x)/2; // abcisse réelle du milieu up
		        float yUpCenter= (pos2d[0].y + pos2d[1].y)/2; // abcisse réelle du milieu up

		        //p.point(xUpCenter, yUpCenter);  // affiche le point       
		        if (debugIn)PApplet.println ("Milieu bord sup : x="+ xUpCenter + " | y=" + yUpCenter ); 
		        
		          // mémorise les valeurs dans les champs de l'objet Marker
		          markersIn[i].upCenter2D.x=(int)xUpCenter;
		          markersIn[i].upCenter2D.y=(int)yUpCenter;

		        //--- calcul du milieu du bord inf du marker 2D --- 
		        float xDownCenter= (pos2d[2].x + pos2d[3].x)/2; // abcisse réelle 
		        float yDownCenter= (pos2d[2].y + pos2d[3].y)/2; // ordonnée réelle 

		        //p.point(xDownCenter, yDownCenter);  // affiche le point       
		        if (debugIn)PApplet.println ("Milieu bord inf : x="+ xDownCenter + " | y=" + yDownCenter ); 
		        
		          // mémorise les valeurs dans les champs de l'objet Marker
		          markersIn[i].downCenter2D.x=(int)xDownCenter;
		          markersIn[i].downCenter2D.y=(int)yDownCenter;

		        //--- calcul du milieu du bord gauche du marker 2D --- 
		        float xLeftCenter= (pos2d[0].x + pos2d[3].x)/2; // abcisse réelle 
		        float yLeftCenter= (pos2d[0].y + pos2d[3].y)/2; // ordonnéeréelle 

		        //p.point(xLeftCenter, yLeftCenter);  // affiche le point       
		        if (debugIn) PApplet.println ("Milieu bord gauche : x="+ xLeftCenter + " | y=" + yLeftCenter );
		        
		          // mémorise les valeurs dans les champs de l'objet Marker
		          markersIn[i].leftCenter2D.x=(int)xLeftCenter;
		          markersIn[i].leftCenter2D.y=(int)yLeftCenter;


		        //--- calcul du milieu du bord droit du marker 2D --- 
		        float xRightCenter= (pos2d[1].x + pos2d[2].x)/2; // abcisse réelle 
		        float yRightCenter= (pos2d[1].y + pos2d[2].y)/2; // ordonnée réelle 

		        //p.point(xRightCenter, yRightCenter);  // affiche le point       
		        if (debugIn) PApplet.println ("Milieu bord droit : x="+ xRightCenter + " | y=" + yRightCenter );
		        
		          // mémorise les valeurs dans les champs de l'objet Marker
		          markersIn[i].rightCenter2D.x=(int)xRightCenter;
		          markersIn[i].rightCenter2D.y=(int)yRightCenter;


		        //--- calcul du centre du marker 2D --- 
		        float xCenter= (xLeftCenter+xRightCenter)/2; // abcisse réelle du centre 
		        float yCenter= (yUpCenter+yDownCenter)/2; // ordonnée réelle du centre 

		        //p.point(xCenter, yCenter);  // affiche le point       
		        if (debugIn) PApplet.println ("Centre : x="+ xCenter + " | y=" + yCenter ); 

		        //p.stroke(255,0,255); 
		        //p.noFill(); 
		        //p.ellipse(xCenter, yCenter,10,10);  // affiche le cercle
		        
		        // -- met à jour et mémorise les coordonnées du centre du Marker --
		        markersIn[i].center2D.x=(int)xCenter;
		        markersIn[i].center2D.y=(int) yCenter;
		        
		        //----- calcul hauteur et largeur MEDIANE du Marker2D --- 
		          float widthMarker= PApplet.abs(xLeftCenter-xRightCenter) ; // prend mediane horizontale  comme référence largeur
		          float heightMarker= PApplet.abs(yUpCenter-yDownCenter) ; // prend mediane verticale comme référence hauteur
		          
		          if (debugIn)  PApplet.println ("hauteur Marker = " + heightMarker); 
		          if (debugIn) PApplet.println ("largeur Marker= " + widthMarker); 
		          
			        // -- met à jour et mémorise les coordonnées du centre du Marker --
		          markersIn[i].height2D=heightMarker; 
		          markersIn[i].width2D=widthMarker; 
		       
		       //------ calcul de l'angle dans l'axe Y du plan 3D du marker
		          
		       PMatrix3D mySyst3D = nyaIn.getMarkerMatrix(i); // récupère la matrice du système de coordonnées 3D du marker... 
		       
		       //------ on récupère l'angle dans l'axe Y à partir de l'élément m02 de la matrice 4x4 ++++
		       // l'élément m02 est insensible à la rotation dans les autres axes si elle existe !! 
		       
		       float angle=PApplet.degrees(PApplet.asin(mySyst3D.m02));
		       
		       markersIn[i].angleAxeY=angle;
		       
		       if (debugIn)  PApplet.println ( "L'angle du plan du marker par rapport à la caméra vaut (par asin de m02) : " 
		    		   +  markersIn[i].angleAxeY + "degrés");  

		       
	        
		      } // fin if numMarkers 
		    
		
		} // fin updateMarkers
		
		//---------------------- fonction de tracé des markers nya détectés ---------- 
		public void draw2DMarkers(MultiMarker nyaIn, Marker[] markersIn, int xRefIn, int yRefIn, float scaleIn, int radius, int colorStrokeIn, int strokeWeightIn, boolean fillIn, int colorFillIn, boolean debugIn) {
			//-- reçoit l'objet Multimarker utilisé pour la détection et le tableau d'objet Marker associé
			// -- chaque objet Marker comporte les champs de description associés au Marker 

			int numMarkers=markersIn.length; // mémorise le nombre marker
			
			// paramètres affichage texte 
		    //p.textAlign(PApplet.LEFT, PApplet.TOP); // paramètre d'affichage du texte 
		    // p.textSize(10); // taille à utiliser pour le texte 
		    
		    // --- paramètre graphique 
		    //p.noStroke();
			
	          //---------- fixe les paramètres graphiques à utiliser -----------
	          if (fillIn) p.fill(colorFillIn) ; else p.noFill(); 
	          p.stroke(colorStrokeIn); // couleur pourtour
	          p.strokeWeight(strokeWeightIn); // épaisseur pourtour 
	          

		    // passe en revue les markers mémorisés dans le Multimarker nya 
	          
		    for (int i=0; i<numMarkers; i++) { // passe en revue les markers de référence 

		      // if the marker does NOT exist (the ! exlamation mark negates it) continue to the next marker, aka do nothing
		      if ((!nyaIn.isExistMarker(i))) { continue; } 
		      
		      // passe au marker suivant si le marker(i) n'est pas détecté

		      // the following code is only reached and run if the marker DOES EXIST
		       
		      	// affiche les coins -- 
		        for (int j=0; j<markersIn[i].corners2D.length; j++) {

		        //p.stroke(255,255,0); 
		        //p.point(markersIn[i].corners2D[j].x, markersIn[i].corners2D[j].y);
		        p.ellipse(xRefIn+(markersIn[i].corners2D[j].x*scaleIn), yRefIn+(markersIn[i].corners2D[j].y*scaleIn), radius, radius);
		        	
		       //PApplet.println ("Coin " + j + " : x="+ pos2d[j].x + " | y=" + pos2d[j].y ); 
		          


		        } // fin for markersIn

		        // --- affiche les milieux des côtés ---
		        p.ellipse(xRefIn+(markersIn[i].upCenter2D.x*scaleIn), yRefIn+(markersIn[i].upCenter2D.y*scaleIn), radius, radius);
		        p.ellipse(xRefIn+(markersIn[i].downCenter2D.x*scaleIn), yRefIn+(markersIn[i].downCenter2D.y*scaleIn), radius, radius);
		        p.ellipse(xRefIn+(markersIn[i].leftCenter2D.x*scaleIn), yRefIn+(markersIn[i].leftCenter2D.y*scaleIn), radius, radius);
		        p.ellipse(xRefIn+(markersIn[i].rightCenter2D.x*scaleIn), yRefIn+(markersIn[i].rightCenter2D.y*scaleIn), radius, radius);
		        
		        /*
		        p.point(markersIn[i].upCenter2D.x, markersIn[i].upCenter2D.y);
		        p.point(markersIn[i].downCenter2D.x, markersIn[i].downCenter2D.y);
		        p.point(markersIn[i].leftCenter2D.x, markersIn[i].leftCenter2D.y);
		        p.point(markersIn[i].rightCenter2D.x, markersIn[i].rightCenter2D.y);
	        	*/
		        
		        // affiche le centre 
		        //p.stroke(0,255,0); 
		        //p.noFill(); 
		        //p.ellipse( markersIn[i].center2D.x, markersIn[i].center2D.y,10,10);  // affiche le cercle
		        p.ellipse(xRefIn+(markersIn[i].center2D.x*scaleIn), yRefIn+(markersIn[i].center2D.y*scaleIn), radius, radius);
	        

		        
		      } // fin if numMarkers 
		
		} // fin draw2DMarkers 
		
		// forme minimal draw2DMarkers
		public void draw2DMarkers(MultiMarker nyaIn, Marker[] markersIn) {

			draw2DMarkers(nyaIn, markersIn, 0, 0, 1, 1, p.color(255,255,0), 1, false, 0, false);
		
		} // fin draw2DMarkers
		
		//---- fonction de dessin 3D du marker --- 
		public void draw3DMarkers(MultiMarker nyaIn, Marker[] markersIn, int widthBoxIn, int heightBoxIn, int depthBoxIn, boolean strokeIn, int colorStrokeIn, int strokeWeightIn, boolean fillIn, int colorFillIn, boolean debugIn) {
			
			int numMarkers=markersIn.length; // mémorise le nombre marker
			
			  //------- OPENGL ou P3D DOIT ETRE ACTIVE ++ cf size() dans setup() ----------- 
			  
			  nyaIn.setARPerspective(); // uniformise la perspective pour tous les markers... 
			  // fixe les paramètres de projection de perspective - cf perspective() - fonction de Processing
			  
			  		  
			  // objets qui disparaissent... cf http://processing.org/reference/frustum_.html
			  // et aussi : http://processing.org/reference/perspective_.html 
			  // -- cf pas avec P3D au lieu OPENGPl


			  for (int i=0; i<numMarkers; i++) { // passe en revue les Markers de référence
				  

			      if ((!nyaIn.isExistMarker(i))) { continue; } // si le marker n'est pas détecté on passe au suivant

			      PMatrix3D markerSyst3D = nyaIn.getMarkerMatrix(i); // récupère le système de coordonnées 3D du marker dans une matrice 4x4 (objet PMatrix3D)... 
			      
			      p.setMatrix(markerSyst3D); // fixe le système de coordonnées du nouveau système de coordonnées
			      
			      if (debugIn) printPMatrix3D(markerSyst3D); // affiche la matrice 3D du marker
			      
			      p.scale(1, -1); // tourne le système de coordonnées pour travailler intuitivement pour les utilisateurs Processing

			      p.scale((float)1.0); // 1 pour taille x1 

			      //drawCurrentSyst3D (int sizeIn,  int strokeXIn, int strokeYIn, int strokeZIn, int strokeWeightIn)
			      drawCurrentSyst3D(50, p.color(255,0,0), p.color(0,255,0), p.color(0,0,255),2); // affiche le repère 0xyz courant avec taille et épaisseur voulus 

			      //--- dessin 3D --- 

			      //lights(); // allume la lampe... 
			      
			     //----- paramètres graphiques
			      if (strokeIn)  {
			    	  p.stroke(colorStrokeIn);
			    	  p.strokeWeight(strokeWeightIn);
			      }
			      else p.noStroke(); // pas de pourtour 
			       

			      if (fillIn) p.fill(colorFillIn,160);  // 2ème valeur = transparence      
			      else p.noFill(); 
			       
			      //--- dessin du plan 3D associé au marker -- 
			      if ((widthBoxIn==0) && (heightBoxIn==0)) { // si les paramètres largeur/hauteur =0, on utilise les paramètres du marker[i]
			    	  p.box (markersIn[i].width3D , markersIn[i].height3D, markersIn[i].depth3D); // plan du marker virtuel avec les propriétés 3D associées au marker
			      }
			      else p.box (widthBoxIn,widthBoxIn, depthBoxIn); // plan du marker -- cf Ok avec P3D mais disparaît avec OPENGL
			      
			      //noLights(); // stop lumières
			  

			  } // fin for numMarkers
			
			    // restaure la perspective par défaut - fonction Processing core.PGraphics
			    p.perspective(); // Calls perspective() with Processing's standard coordinate projection. 

		} // fin draw3DMarkers

		
		//--------- calcul de la distance du Marker ---------------- 
		public float distanceMarker (Marker markerIn, float ouvertureIn, int widthCaptureIn, boolean debugIn) {
			
			 float distanceOut=(float)0.0; 
			
	          //---- calcul distance à partir taille en pixel
	          
	          //Float pixelsToAngle(Float pixelSizeIn, Float ouvertureIn, Float widthCaptureIn)
	          float angleApparentMarker= pixelsToAngle(markerIn.height2D, ouvertureIn, (float) widthCaptureIn, debugIn ); // calcul l'angle apparent correspondant en se basant sur hauteur 
	          // on se base sur la hauteur car elle ne varie pas selon orientation si le marqueur vertical (sur un mur par exemple) 
	          
	          // Float realDistance( Float angleAppDegIn, Float sizeMarkerIn)
	          distanceOut=realDistance (angleApparentMarker, markerIn.realWidth/10, debugIn); // l'unité widthMarker idem distance calculée - affiche auto... à corriger pour récup variable... 
	          
	          return (distanceOut); 
	          
		}// fin distance Marker
		
		//-------- fonction de calcul de la distance pour un tableau de Markers ------- 
		public void distanceMarkers(MultiMarker nyaIn, Marker[] markersIn, float ouvertureIn, int widthCaptureIn, boolean debugIn) {
			
			//-- reçoit l'objet Multimarker utilisé pour la détection et le tableau d'objet Marker associé
			// -- chaque objet Marker comporte les champs de description associés au Marker 

			int numMarkers=markersIn.length; // mémorise le nombre marker
			
		    // passe les markers en revue 
			
		    for (int i=0; i<numMarkers; i++) { // passe en revue les markers de référence 

		      // si le marker n'existe pas = càd n'a été détecté, alors on passe à l'indice suivant 
		      if ((!nyaIn.isExistMarker(i))) { continue; } // passe au marker suivant si le marker(i) n'est pas détecté

		      // le code qui suit n'est exécuté que si le marker "existe", c'est à dire uniquement si le marker a été détecté
		       
		       // calcul la distance du Marker par rapport à la camera 
		      markersIn[i].distance =distanceMarker(markersIn[i],ouvertureIn, widthCaptureIn, debugIn);
		      if (debugIn) PApplet.println("Distance du marker " + markersIn[i].name + "="+ markersIn[i].distance + " cm.");

		        
		      } // fin if numMarkers 
		
			
		} // fin distance Markers 

		//-------- fonction de test existence Marker à partir indice ------- 
		public boolean isExistMarker(MultiMarker nyaIn, int indiceIn) {
			
			return (nyaIn.isExistMarker(indiceIn)); 
			
		}

		//-------- fonction de test existence Marker à partir chaine nom  ------- 
		public boolean isExistMarker(MultiMarker nyaIn, Marker[] markersIn, String nameIn) {
			
			//-- reçoit l'objet Multimarker utilisé pour la détection et le tableau d'objet Marker associé
			// -- chaque objet Marker comporte les champs de description associés au Marker 

			int numMarkers=markersIn.length; // mémorise le nombre marker
			
			int numSelectedMarker=-1; 
			
		    // passe les markers en revue 
			
		    for (int i=0; i<numMarkers; i++) { // passe en revue les markers de référence 

		    	//if (markersIn[i].name==nameIn) numSelectedMarker=i;
		    	if (PApplet.match(markersIn[i].name,nameIn)!=null) numSelectedMarker=i; 
		        
		      } // fin if numMarkers 
			
			if (numSelectedMarker>=0) {
				return (nyaIn.isExistMarker(numSelectedMarker)); // teste si le marker est détecté
			}
			else return (false); // si numSelected = -1 = pas de marker de ce nom 
			
		}

		//-------- fonction de test existence Marker à partir de son numero (pas indice)  ------- 
		public boolean isExistMarker(MultiMarker nyaIn, Marker[] markersIn, int numMarkerIn) {
			
			//-- reçoit l'objet Multimarker utilisé pour la détection et le tableau d'objet Marker associé
			// -- chaque objet Marker comporte les champs de description associés au Marker 

			String nameMarker=""; 
			
			/* --- non ! pas nécessaire 
			if (numMarkerIn<10)nameMarker="4x4_00"+numMarkerIn+".patt";
			if (numMarkerIn<100)nameMarker="4x4_0"+numMarkerIn+".patt";
			if (numMarkerIn<1000)nameMarker="4x4_"+numMarkerIn+".patt";
			*/
			
			nameMarker="4x4_"+numMarkerIn+".patt";
			
			// PApplet.println("nameMarker =" +nameMarker); // debug
			
			return (isExistMarker(nyaIn, markersIn, nameMarker));
			
			
		}
		
		//-------- fonction de sélection d'un Marker à partir chaine nom  ------- 
		public Marker selectMarker(Marker[] markersIn, String nameIn) {
			
			//-- reçoit l'objet Multimarker utilisé pour la détection et le tableau d'objet Marker associé
			// -- chaque objet Marker comporte les champs de description associés au Marker 

			int numMarkers=markersIn.length; // mémorise le nombre marker
			
			int numSelectedMarker=-1; 
			
		    // passe les markers en revue 
			
		    for (int i=0; i<numMarkers; i++) { // passe en revue les markers de référence 

		    	//if (markersIn[i].name==nameIn) numSelectedMarker=i;
		    	if (PApplet.match(markersIn[i].name,nameIn)!=null) numSelectedMarker=i; // si le nom correspond 
		        
		      } // fin if numMarkers 
			
			if (numSelectedMarker>=0) {
				return (markersIn[numSelectedMarker]); // renvoie le marker correspondant 
			}
			else return (null); // si numSelected = -1 = pas de marker de ce nom 
			
		}
		
		//-------- fonction de sélection d'un Marker à partir de son numero (pas indice)  ------- 
		public Marker selectMarker(Marker[] markersIn, int numMarkerIn) {
			
			//-- reçoit l'objet Multimarker utilisé pour la détection et le tableau d'objet Marker associé
			// -- chaque objet Marker comporte les champs de description associés au Marker 

			String nameMarker=""; 
			
			/* --- non ! pas nécessaire 
			if (numMarkerIn<10)nameMarker="4x4_00"+numMarkerIn+".patt";
			if (numMarkerIn<100)nameMarker="4x4_0"+numMarkerIn+".patt";
			if (numMarkerIn<1000)nameMarker="4x4_"+numMarkerIn+".patt";
			*/
			
			nameMarker="4x4_"+numMarkerIn+".patt";
			
			// PApplet.println("nameMarker =" +nameMarker); // debug
			
			return (selectMarker(markersIn,nameMarker));
			
			
		}
		
		/////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////// FONCTIONS DESSIN PROCESSING UTILES ///////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////
		
		public void drawShape(Point[] pointIn, int xRefIn, int yRefIn, float scaleIn, int colorStrokeIn, int strokeWeightIn, boolean fillIn, int colorFillIn, boolean debug){
			
			// cette fonction dessine une forme fermée à partir d'un tableau de points 
	
			//------- paramètres graphiques -------- 
	          //---------- fixe les paramètres graphiques à utiliser -----------
	          if (fillIn) p.fill(colorFillIn) ; else p.noFill(); 
	          p.stroke(colorStrokeIn); // couleur pourtour
	          p.strokeWeight(strokeWeightIn); // épaisseur pourtour 
			
			//---------- tracé de la forme ----- 
			   
			   p.beginShape();

			   for (int n=0; n<pointIn.length; n++) { // défile les points 
				 if (debug) PApplet.println ("Point " + n + " :  x= "+ pointIn[n].x +" | y= " + pointIn[n].y ); // affiche les coordonnées du point 
			     p.vertex(xRefIn +(pointIn[n].x*scaleIn), yRefIn+(pointIn[n].y*scaleIn));
			     
			   } // fin for
			   
			   p.endShape(PApplet.CLOSE);
	
		}
		
		
		
		///////////////// fonctions internes diverses ////////////////////::
		
		//--- selectBuffer : renvoie le Buffer sélectionné à partir de la chaine de caractère reçue
		
		opencv_core.IplImage selectBuffer (String stringBufferIn) { // reçoit chaine et renvoie IplIMage buffer 
			
			if (stringBufferIn=="BUFFER") return(Buffer); 
			else if (stringBufferIn=="GRAY") return(BufferGray); 
			else if (stringBufferIn=="RED") return(BufferR); 
			else if (stringBufferIn=="GREEN") return(BufferG); 
			else if (stringBufferIn=="BLUE") return(BufferB); 
			else if (stringBufferIn=="MEMORY") return(Memory); 
			else if (stringBufferIn=="MEMORY2") return(Memory2); 
			else return(Buffer); // si la chaine n'est pas reconnue, on renvoie le Buffer principal 
		
		} // fin fonction sélectBuffer
		


} // fin de la classe OpenCV
