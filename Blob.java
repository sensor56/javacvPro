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

public class Blob {
	
	/*
	 * La classe Blob crée un objet correspondant à un contour unitaire
	 * constitué d'un tableau de points
	 * caractérisé par sa longueur, son nombre de point, son aire, son point central
	 * 
	 * Cette classe est utilisée pour transférer les données issues d'un CvSeq vers Processing
	 * 
	 */
	////////////// VARIABLES ////////////////
	// toutes les variables peuvent être private ou public
	
	//--- variables de d'instances (non static) ---
	// accessible pour une seule instance
	// à déclarer en private et accéder par accesseur
	// il n'est pas directement accessible. 
	
	public int indiceContour; // indice du contour dans le CvSeq de détection global - usage interne 
	
    public float area=0; // aire du blob
    public float lengthArc=0; // le périmètre (longueur)  du blob
    
    public float length=0; // le nombre de points du blob
    
    public Point centroid		= new Point(); // le point du centre du Blob
    
    public Point[] points		= new Point[0]; // le tableau de points du blob
	
    public Rectangle rectangle	= new Rectangle(); // le rectangle entourant le Blob	

		
	
	///////////   CONSTRUCTEURS  //////////
	// doivent avoir obligatoirement le meme nom que la classe
	
	//--- le constructeur par défaut - obligatoire
	protected Blob(){ 
		// ce constructeur est "protected" est n'est pas accessible directement de l'extérieur

	}


	protected Blob( int indiceContourIn, float areaIn, float lengthArcIn, float lengthIn, Point centroidIn, Rectangle rectIn, Point[] pointsIn){ 
		// c'est ce constructeur blob qui est utilisé par la fonction blobs() de la classe Opencv
		// ce constructeur est "protected" est n'est pas accessible directement de l'extérieur
		
		
			this.indiceContour= indiceContourIn;
			this.area		= areaIn; // l'aire du Blob
			this.lengthArc		= lengthArcIn; // le périmètre du blob
			
			this.length	= lengthIn; // nombre de points 
			this.centroid	= centroidIn; // le centre du blob
			
			this.rectangle	= rectIn; // le rectangle entourant
			
			this.points		= pointsIn; // le tableau de points du blob
			
		
	} // fin constructeur Blob

	
	///////////////////// METHODES ////////////////////////////
	
	// NB : les méthodes n'utilisant que des variables de classe doivent déclarées static
	
	//---- méthodes accesseurs (get) ---- (création automatique possible) 
	//--- pour accéder aux variables d'instance et aux variables de classe 
	
	//----- méthodes mutateurs (set) ---- (création automatique possible) 
	//---- pour modifier les variables d'instance et les variables de classe
	
	//---- méthodes de classe ------
	

} // fin de la classe blob 

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