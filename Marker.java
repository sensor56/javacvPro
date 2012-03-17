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


public class Marker {

	/*
	 * La classe Marker crée un objet correspondant à un marker détecté avec la librairie nyar4psg (Artoolkit) : 
	 * 
	 * 
	 * name : nom du fichier description à utilisé *.patt
	 * 
	 */

	
	////////////// VARIABLES ////////////////
	// toutes les variables peuvent être private ou public
	
	//--- variables de d'instances (non static) ---
	// accessible pour une seule instance
	// à déclarer en private et accéder par accesseur


	 public String name		="" ; // le nom du fichier de description du marker

	 public float realX		= (float) 0.0; // abscisse réelle au sol de l'espace d'évolution décrit par les markers
	 public float realY		= (float) 0.0; // ordonnée réel au sol de l'espace d'évolution décrit par les markers
	 
	 public float realWidth		= (float) 0.0; // largeur réelle du Marker

	 public float width2D		= (float) 0.0; // largeur 2D du Marker telle que affichée sur l'image webcam
	 public float height2D		= (float) 0.0; // hauteur 2D du Marker telle que affichée sur l'image webcam

	 //--- pour réalité augmentée -- 
	 public float width3D		= (float) 1000; // largeur 3D du Marker telle que affichée sur l'image webcam
	 public float height3D		= (float) 1000; // hauteur 3D du Marker telle que affichée sur l'image webcam
	 public float depth3D		= (float) 10; // profondeur 3D du Marker telle que affichée sur l'image webcam

	 public float distance		= (float) 0.0; // distance du marker à la webcam (calculée)

	 public float angleAxeY		= (float) 0.0; // angle de rotation dans l'axe Y en degrés

	 
	 public Point upCenter2D = new Point(); // milieu bord sup 2D du marqueur
	 public Point downCenter2D = new Point(); // milieu bord inf 2D du marqueur
	 public Point leftCenter2D = new Point(); // milieu bord gauche 2D du marqueur
	 public Point rightCenter2D = new Point(); // milieu bord droit 2D du marqueur

	 public Point center2D = new Point(); // centre 2D du marqueur

	 public Point[] corners2D = new Point[4]; // coins 2D du marqueur
	 
	//--- variables de classe (static)
	// = accessible pour toutes les instances
	// à déclarer en public ou private 
	
		
	
	///////////   CONSTRUCTEURS  //////////
	// doivent avoir obligatoirement le meme nom que la classe
	
	//--- le constructeur par défaut
	public Marker (){
	
		//---- initialise les points de coins
		this.corners2D[0]=new Point(); // coin sup gauche
		this.corners2D[1]=new Point(); // coin sup droite
		this.corners2D[2]=new Point(); // coin inf droite
		this.corners2D[3]=new Point(); // coin inf gauche
		
	}
	
/*
	protected Marker (Point centerIn, float radiusIn) {

		// ce constructeur est utilisé par des fonctions utilisant/renvoyant des cercles 

		this.center=centerIn; 
		this.radius=radiusIn;
		
	}
*/
	///////////////////// METHODES ////////////////////////////
	
	// NB : les méthodes n'utilisant que des variables de classe doivent déclarées static
	
	//---- méthodes accesseurs (get) ---- (création automatique possible) 
	//--- pour accéder aux variables d'instance et aux variables de classe 
	
	//----- méthodes mutateurs (set) ---- (création automatique possible) 
	//---- pour modifier les variables d'instance et les variables de classe
	
	//---- méthodes de classe ------


}