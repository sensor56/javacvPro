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


public class Circle {

	/*
	 * La classe Circle crée un objet correspondant à un cercle associant : 
	 * 
	 * 
	 * centre : point de corrdonnées x,y
	 * radius : rayon du cercle(float)
	 * 
	 */

	
	////////////// VARIABLES ////////////////
	// toutes les variables peuvent être private ou public
	
	//--- variables de d'instances (non static) ---
	// accessible pour une seule instance
	// à déclarer en private et accéder par accesseur


	 public Point center		= new Point(); // le point du centre du cercle 
 
	 public float radius		= 0; // la valeur du rayon du cercle
	 
	//--- variables de classe (static)
	// = accessible pour toutes les instances
	// à déclarer en public ou private 
	
		
	
	///////////   CONSTRUCTEURS  //////////
	// doivent avoir obligatoirement le meme nom que la classe
	
	//--- le constructeur par défaut
	protected Circle (){
	
		
	}
	

	protected Circle (Point centerIn, float radiusIn) {

		// ce constructeur est utilisé par des fonctions utilisant/renvoyant des cercles 

		this.center=centerIn; 
		this.radius=radiusIn;
		
	}

	///////////////////// METHODES ////////////////////////////
	
	// NB : les méthodes n'utilisant que des variables de classe doivent déclarées static
	
	//---- méthodes accesseurs (get) ---- (création automatique possible) 
	//--- pour accéder aux variables d'instance et aux variables de classe 
	
	//----- méthodes mutateurs (set) ---- (création automatique possible) 
	//---- pour modifier les variables d'instance et les variables de classe
	
	//---- méthodes de classe ------


}
