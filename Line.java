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

public class Line extends OpenCV { // extends OpenCV pour avoir accès aux fonctions OpenCV 
	/*
	 * La classe Line crée un objet correspondant à une droite associant : 
	 * 
	 * soit : 
	 * point1 : point de corrdonnées x1,y1
	 * point2 : point de corrdonnées x2,y2
	 * 
	 * soit : 
	 * un vecteur 1 à l'origine définit par 2 valeurs float theta1 (angle avec la verticale) et rho1 (distance à l'origine)
	 * un vecteur 2 à l'origine définit par 2 valeurs float theta2 (angle avec la verticale) et rho2 (distance à l'origine)
	 * 
	 */

	
	////////////// VARIABLES ////////////////
	// toutes les variables peuvent être private ou public
	
	//--- variables d'instances (non static) ---
	// accessible pour une seule instance
	// à déclarer en private et accéder par accesseur - laisser public si accès direct externe souhaité

	// -- une droite peut être définie par 2 points -- 
	public Point point1		= new Point(); // le point 1
	public Point point2		= new Point(); // le point 2 
 
	 // -- une droite peut aussi être définie par un vecteur perpendiculaire à la droite et passant par l'origine (vecteur normal)
	public float theta		= 0; // angle du vecteur perpendiculaire à la droite et passant par l'origine
	public float rho		= 0; // rayon du vecteur perpendiculaire à la droite et passant par l'origine
	 
	 //---- variables calculées ---
	 
	 // l'équation de la droite peut s'écrit classiquement y=ax+b 
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
	protected Line (){
	
		
	}
	

	//------- constructeur en coordonnées polaires --------- 
	//protected Line (float thetaIn, float rhoIn) {
	public Line (float thetaIn, float rhoIn, float YMaxIn) {
		// ce constructeur est utilisé par des fonctions utilisant/renvoyant des droites ( hough) 
		// en les définissant à partir du vecteur normal à la droite passant par l'origine

		// thetaIn : angle du vecteur perpendiculaire à la droite et passant par l'origine
		// rhoIn : rayon du vecteur perpendiculaire à la droite et passant par l'origine
		// YMax : valeur à utiliser pour calcul pointYMax - typiquement heightImage
		
		//--- vecteur normal à la droite et passant par l'origine 
		this.theta=thetaIn; // angle
		this.rho=rhoIn; // longueur
		
		//---- calcul du point croisant l'axe des ordonnées x=0 à partir de rho et theta        
        //---- calcul y0 pour x=0 --- 
        // cf dans triangle rect = hyp=adj/cos() = rho / cos((pi/2)-theta)
        float y0=rho/PApplet.cos(PApplet.HALF_PI-theta); 
        this.pointX0.x=0; // mémorise coordonnées - intersection avec axe ordonnées (x=0)
        this.pointX0.y=(int)y0; // mémorise coordonnées 

        PApplet.println("Point X0 coupant axe des ordonnées : (" +pointX0.x+ ", "+pointX0.y+")"); // debug
                
		//---- calcul du point croisant l'axe des abcisses y=0 à partir de rho et theta 
	     //---- calcul x0 pour y=0 --- 
        // cf dans triangle rect = hyp=adj/cos() = rho / cos(theta)
        float x0=rho/PApplet.cos(theta); 
        this.pointY0.x=(int)x0; // mémorise coordonnées
        this.pointY0.y=0; // mémorise coordonnées - intersection avec axe abscisses (y=0)

        PApplet.println("Point Y0 coupant axe des abscisses : (" +pointY0.x+ ", "+pointY0.y+")"); // debug
   
        //--- calcul des paramètres a et b de l'équation y=ax+b à partir des 2 points connus de la droite
        
        float[] ab= calculEquationDroite(pointX0, pointY0); // apppelle fonction de la classe OpenCV 
        
        a=ab[0]; // récupère valeur de a
        b=ab[1]; // récupère valeur de b
	
        PApplet.println("a ="+ a + " | b=" + b);// debug 
		
        //------ calcul du point d'intersection de la droite avec l'axe y=heightCapture 
        //Point myPoint=interLines(0,height(), a, b);
        pointYMax=interLines(0,YMaxIn, a, b);
        
        //PApplet.println("Point yMax : x= " + pointYMax.x + "| y = " + pointYMax.y); 

	} // fin Line 
	
	//----------- constructeur en coordonnées cartésiennes ------
	public Line (Point point1In, Point point2In,float YMaxIn) {

		// ce constructeur est utilisé par des fonctions utilisant/renvoyant des droite ( hough)  

		this.point1=point1In; // définition premier point
		this.point2=point2In; // définition deuxième point
		
      //--- calcul des paramètres a et b de l'équation y=ax+b à partir des 2 points connus de la droite
        
        float[] ab= calculEquationDroite(point1, point2); // apppelle fonction de la classe OpenCV 
        
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
