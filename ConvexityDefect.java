package monclubelec.javacvPro;


//======= import =========
//--- processing --- 
import processing.core.*;

//--- javacv / javacpp ---- 
import com.googlecode.javacpp.*;
import com.googlecode.javacpp.Loader;
import com.googlecode.javacv.*;
import com.googlecode.javacv.cpp.opencv_core;
import com.googlecode.javacv.cpp.opencv_imgproc;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

//---- java ---- 
import java.nio.*; // pour classe ByteBuffer
import java.awt.*; // pour classes Point , Rectangle..


public class ConvexityDefect {
	
	/*
	 * La classe ConvexityDefect crée un objet correspondant à un creux d'un contour
	 * constitué d'un points de début
	 * d'un point de fin
	 * d'un point de profondeur maximale
	 * d'un double correspondant à la profondeur maximale
	 * 
	 * Cette classe est utilisée pour transférer les données issues d'un CvSeq vers Processing
	 * 
	 */

	
	////////////// VARIABLES ////////////////
	// toutes les variables peuvent être private ou public
	
	//--- variables de d'instances (non static) ---
	// accessible pour une seule instance
	// à déclarer en private et accéder par accesseur

	 //public int indiceBlob; // l'indice du Blob auquel appartient le ConvexityDefect

	 public Point start		= new Point(); // le point de début du ConvexityDefect
	 public Point end		= new Point(); // le point de fin du ConvexityDefect
	 public Point depth		= new Point(); // le point de profondeur maximale du ConvexityDefect
	 
	 public float value		= 0; // la valeur de la profondeur maximale du ConvexityDefect
	 
	 //--- variables calculées --- 
	 public float distSE= 0; // la valeur de la distance entre les points Start et End
	 public float distSD= 0; // la valeur de la distance entre les points Start et Depth
	 public float distDE= 0; // la valeur de la distance entre les points Depth et End
	 
	 public float angleSDE		= 0; // la valeur en radians de l'angle Start - Depth - End
	 
	//--- variables de classe (static)
	// = accessible pour toutes les instances
	// à déclarer en public ou private 
	
		
	
	///////////   CONSTRUCTEURS  //////////
	// doivent avoir obligatoirement le meme nom que la classe
	
	//--- le constructeur par défaut
	protected ConvexityDefect (){
	
		
	}
	

	protected ConvexityDefect (Point startIn, Point endIn, Point depthIn, float valueIn, float distSEIn, float distSDIn, float distDEIn, float angleSDEIn){
		// c'est ce constructeur ConvexityDefect qui est utilisé par la fonction convexityDefects() de la classe Opencv
		// ce constructeur est "protected" est n'est pas accessible directement de l'extérieur

		this.start=startIn; 
		this.end=endIn;
		this.depth=depthIn; 		
		this.value=valueIn; 
		
		//--- variables calculées --- 
		this.distSE=distSEIn;
		this.distSD=distSDIn; 
		this.distDE=distDEIn;
		this.angleSDE=angleSDEIn; 
		
	}

	///////////////////// METHODES ////////////////////////////
	
	// NB : les méthodes n'utilisant que des variables de classe doivent déclarées static
	
	//---- méthodes accesseurs (get) ---- (création automatique possible) 
	//--- pour accéder aux variables d'instance et aux variables de classe 
	
	//----- méthodes mutateurs (set) ---- (création automatique possible) 
	//---- pour modifier les variables d'instance et les variables de classe
	
	//---- méthodes de classe ------
	

} // fin de la classe ConvexityDefect




	

/*

/** exemple Javadoc 
* 
* @ param nomvariable description variable 
*
* @return PImage
* 
* 
* After the <code>loadPixels()</code> call, pixels can by accessed via the <code>pixels</code> variable.
* 
* <p>! NOT IMPLEMENTED YET</p>
* 
* @see #nomfonction
*
* 
* @see OpenCV#blobs( int, int, int, boolean )
*  
* @usage Application
* 
*/


/*

Modèle de classe 

	public class NomClasse {
	

	////////////// VARIABLES ////////////////
	// toutes les variables peuvent être private ou public
	
	//--- variables de d'instances (non static) ---
	// accessible pour une seule instance
	// à déclarer en private et accéder par accesseur
	
	//--- variables de classe (static)
	// = accessible pour toutes les instances
	// à déclarer en public ou private 
	
		
	
	///////////   CONSTRUCTEURS  //////////
	// doivent avoir obligatoirement le meme nom que la classe
	
	//--- le constructeur par défaut
	public NomClasse(){
		
	}
	
	
	///////////////////// METHODES ////////////////////////////
	
	// NB : les méthodes n'utilisant que des variables de classe doivent déclarées static
	
	//---- méthodes accesseurs (get) ---- (création automatique possible) 
	//--- pour accéder aux variables d'instance et aux variables de classe 
	
	//----- méthodes mutateurs (set) ---- (création automatique possible) 
	//---- pour modifier les variables d'instance et les variables de classe
	
	//---- méthodes de classe ------
	
	
	} fin de classe 
	

*/