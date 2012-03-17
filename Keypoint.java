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


public class Keypoint {
	
	/*
	 * La classe Keypoint crée un objet correspondant à un point clé associant : 
	 * 
	 * 
	 * Point de corrdonnées x,y
	 * size : rayon associé au point (float)
	 * angle associé au point (float)
	 * octave associé au point (int)
	 * 
	 */

	
	////////////// VARIABLES ////////////////
	// toutes les variables peuvent être private ou public
	
	//--- variables de d'instances (non static) ---
	// accessible pour une seule instance
	// à déclarer en private et accéder par accesseur


	 public Point point		= new Point(); // le point clé
 
	 public float size		= 0; // la valeur du rayon associé
	 public float angle		= 0; // la valeur de l'angle associé
	 public int octave		= 0; // la valeur de l'octave associé
	 
	 
	//--- variables de classe (static)
	// = accessible pour toutes les instances
	// à déclarer en public ou private 
	
		
	
	///////////   CONSTRUCTEURS  //////////
	// doivent avoir obligatoirement le meme nom que la classe
	
	//--- le constructeur par défaut
	protected Keypoint (){
	
		
	}
	

	protected Keypoint (Point pointIn, float sizeIn, float angleIn , int octaveIn) {

		// ce constructeur est utilisé par des fonctions utilisant les points clés (fonction utilisant la fonction native opencv FeatureDetector)
		// ce constructeur est "protected" est n'est pas accessible directement de l'extérieur

		this.point=pointIn; 
		this.size=sizeIn;
		this.angle=angleIn;
		this.octave=octaveIn;
		
	}

	///////////////////// METHODES ////////////////////////////
	
	// NB : les méthodes n'utilisant que des variables de classe doivent déclarées static
	
	//---- méthodes accesseurs (get) ---- (création automatique possible) 
	//--- pour accéder aux variables d'instance et aux variables de classe 
	
	//----- méthodes mutateurs (set) ---- (création automatique possible) 
	//---- pour modifier les variables d'instance et les variables de classe
	
	//---- méthodes de classe ------


}
