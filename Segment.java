package monclubelec.javacvPro;

//======= import =========
//--- processing --- 
import processing.core.*;


//--- javacv / javacpp ---- 
import com.googlecode.javacpp.*;
import com.googlecode.javacv.*;
import com.googlecode.javacv.cpp.opencv_core;
import com.googlecode.javacv.cpp.opencv_imgproc;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

//---- java ---- 
import java.nio.*; // pour classe ByteBuffer
import java.awt.*; // pour classes Point , Rectangle..

public class Segment extends OpenCV { // extends OpenCV pour avoir accès aux fonctions OpenCV 
	/*
	 * La classe Segment crée un objet correspondant à un segment de droite défini par :
	 * 
	 * pointStart : point de corrdonnées xStart, yStart
	 * pointEnd : point de corrdonnées yStart, yEnd
	 * 
	 */

	
	////////////// VARIABLES ////////////////
	// toutes les variables peuvent être private ou public
	
	//--- variables d'instances (non static) ---
	// accessible pour une seule instance
	// à déclarer en private et accéder par accesseur - laisser public si accès direct externe souhaité

	// -- une droite peut être définie par 2 points -- 
	public Point pointStart		= new Point(); // le point de debut
	public Point pointEnd		= new Point(); // le point de fin 

	 //---- variables calculées ---
	 
	 // l'équation de la droite du segment peut s'écrire classiquement y=ax+b 
	public float a=0; // pente de la droite
	public float b=0; // valeur b - = ordonnée à l'origine (pour x=0)
	 
	public Point pointY0= new Point(); // point d'intersection de la droite et axe y=0
	public Point pointX0= new Point(); // point d'intersection de la droite et axe x=0
	 
	public Point pointYMax= new Point(); // point d'intersection de la droite et axe y=heightImage
	 
	//--- variables de classe (static)
	// = accessible pour toutes les instances
	// à déclarer en public ou private 
	
		
	
	///////////   CONSTRUCTEURS  //////////
	// doivent avoir obligatoirement le meme nom que la classe
	
	//--- le constructeur par défaut
	protected Segment (){
	
		
	}
	

	//----------- constructeur en coordonnées cartésiennes ------
	public Segment (Point pointStartIn, Point pointEndIn,float YMaxIn) {

		// ce constructeur est utilisé par des fonctions utilisant/renvoyant des droite ( hough)  

		this.pointStart=pointStartIn; // définition premier point
		this.pointEnd=pointEndIn; // définition deuxième point
		
    //--- calcul des paramètres a et b de l'équation y=ax+b à partir des 2 points connus de la droite
      
      float[] ab= calculEquationDroite(pointStart, pointEnd); // apppelle fonction de la classe OpenCV 
      
      a=ab[0]; // récupère valeur de a
      b=ab[1]; // récupère valeur de b
	
      PApplet.println("a ="+ a + " | b=" + b);// debug 
      
      //------ calcul du point d'intersection de la droite avec l'axe y=heightCapture (donc a=0 et b=0)
      pointYMax=interLines(0,YMaxIn, a, b);

      //------ calcul du point d'intersection de la droite avec l'axe y=0 (donc a=0 et b=0)
      pointY0=interLines(0,0, a, b);
      
      //------ calcul du point d'intersection de la droite avec l'axe x=0 (donc point (0,b))
      pointX0=new Point (0,(int)b);
      

	}
	

	///////////////////// METHODES ////////////////////////////
	
	// NB : les méthodes n'utilisant que des variables de classe doivent déclarées static
	
	//---- méthodes accesseurs (get) ---- (création automatique possible) 
	//--- pour accéder aux variables d'instance et aux variables de classe 
	
	//----- méthodes mutateurs (set) ---- (création automatique possible) 
	//---- pour modifier les variables d'instance et les variables de classe
	
	//---- méthodes de classe ------


	
	
	
}