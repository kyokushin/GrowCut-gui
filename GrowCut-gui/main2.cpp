#include <iostream>
#include <queue>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "opencv_windows_lib.h"
#include "GrowCut.h"

using namespace std;
using namespace cv;


struct MouseEventContainer {
	MouseEventContainer()
		:line_tickness(10),drawing(false),
		drawing_left(false),drawing_right(false)
	{}

	Mat src_img;
	Mat label_img;

	bool drawing_left;
	bool drawing_right;
	bool drawing;
	Point before_pos_list;
	int line_tickness;
	string wname;
	Scalar color;

	Mat gray_label_img;
	Mat mask_img;
	Mat show_img;
};

void genShowImage( MouseEventContainer& mparam ){

	Mat &src_img = mparam.src_img;
	Mat &label_img = mparam.label_img;

	Mat &gray_label_img = mparam.gray_label_img;
	Mat &mask_img = mparam.mask_img;
	Mat &show_img = mparam.show_img;

	cvtColor( label_img, gray_label_img, CV_BGR2GRAY);

	src_img.copyTo( show_img );

	threshold( gray_label_img, mask_img, 10, 255, THRESH_BINARY );
	label_img.copyTo( show_img, mask_img );

}

#define DEBUG_mouse_func
void mouse_func(int event, int x, int y, int flags, void* param){
	const int wait_time = 5;

	MouseEventContainer *mparam = (MouseEventContainer*)param;

	genShowImage( *mparam );
	circle( mparam->show_img, Point( x, y ), 3, Scalar(255,255,255), -1 ); 
	circle( mparam->show_img, Point( x, y ), 2, Scalar(0,0,0), -1 ); 
	imshow( mparam->wname, mparam->show_img );
	waitKey( 5 );

	switch( event ){
	case CV_EVENT_LBUTTONDOWN:
		mparam->drawing_left = true;
		break;

	case CV_EVENT_LBUTTONUP:
		mparam->drawing_left = false;
		break;

	case CV_EVENT_RBUTTONDOWN:
		mparam->drawing_right = true;
		break;

	case CV_EVENT_RBUTTONUP:
		mparam->drawing_right = false;
		break;

	case CV_EVENT_MOUSEMOVE:
		break;

	default:
		break;
	}
	if( mparam->drawing ) return;

	mparam->drawing = true;

	switch( event ){
	case CV_EVENT_LBUTTONDOWN:
#ifdef DEBUG_mouse_func
		cerr<< "Left button down" <<endl;
#endif

		mparam->before_pos_list = Point(x,y);
		mparam->color = Scalar( 0,0,255);
		break;

	case CV_EVENT_LBUTTONUP:
#ifdef DEBUG_mouse_func
		cerr<< "Left button up" <<endl;
#endif
		line(mparam->label_img, mparam->before_pos_list, Point(x,y), mparam->color, mparam->line_tickness );
		genShowImage( *mparam );
		imshow( mparam->wname, mparam->show_img );
		waitKey(wait_time);
		break;

	case CV_EVENT_RBUTTONDOWN:
#ifdef DEBUG_mouse_func
		cerr<< "Right button down" <<endl;
		cerr<< "flag>" << flags <<endl;
#endif
		if( flags & CV_EVENT_FLAG_SHIFTKEY ){
			mparam->before_pos_list = Point(x,y);
			mparam->color = Scalar( 0, 255, 0);
		}
		else {
			mparam->before_pos_list = Point(x,y);
			mparam->color = Scalar( 255,0,0 );
		}
		break;

	case CV_EVENT_RBUTTONUP:
#ifdef DEBUG_mouse_func
		cerr<< "Right button up" <<endl;
#endif
		line(mparam->label_img, mparam->before_pos_list, Point(x,y), mparam->color, mparam->line_tickness );
		genShowImage( *mparam );
		imshow( mparam->wname, mparam->show_img );
		waitKey( wait_time );
		break;
	}

	if( mparam->drawing_left || mparam->drawing_right ){
#ifdef DEBUG_mouse_func
		cerr<< "mouse move and draw ... " <<flush;
#endif

		
		line(mparam->label_img, mparam->before_pos_list, Point(x,y), mparam->color, mparam->line_tickness );
		mparam->before_pos_list = Point(x,y);

		genShowImage( *mparam );
		imshow( mparam->wname, mparam->show_img );
		waitKey( wait_time );
#ifdef DEBUG_mouse_func
		cerr<< "finish" <<endl;
#endif

	}
	
	mparam->drawing = false;

}


void growCut( Mat& src_image, Mat& label_image, Mat& dst ){

	//ÉâÉxÉãÇÃê∂ê¨
	Mat label( label_image.size(), CV_8UC1 );

	for( int i=0; i<label_image.rows; i++ ){
		unsigned char* line_image = (unsigned char*)label_image.ptr(i);
		unsigned char* line_label = (unsigned char*)label.ptr(i);

		for( int h=0; h<label_image.cols; h++ ){
			unsigned char *pix = line_image + 3 * h;
			unsigned char *pix_label = line_label + h;

			if( pix[2] == 255 ){
				*pix_label = 1;
				//cout<< 'b' <<endl;
			}
			else if( pix[0] == 255 ){
				//cout<< 'r' <<endl;
				*pix_label = 2;
			}
			else if( pix[1] == 255 ){
				*pix_label = 3;
			}
			else{
				*pix_label = 0;
				//cout<< "" <<endl;
			}
		}
	}
	//cout<< label <<endl;

	Mat strength( label_image.size(), CV_64F );
	//strength.setTo( Scalar(0) );
	
	for( int i=0; i<strength.rows; i++ ){
		for( int h=0; h<strength.cols; h++ ){
			unsigned char label_val = label.at<unsigned char>(i,h);
			if( label_val == 0 ){
				strength.at<double>( i, h ) = 0.0;
			}
			else strength.at<double>( i, h ) = 1.0;
		}
	}
	
	Mat next_strength = strength.clone();
	Mat next_label = label.clone();

	ys::GrowCut _growCut( src_image, label );
	
	dst.create( src_image.size(), CV_8UC3);
	int converged = 0;
	int itr_count = 0;
	do{
		
		_growCut.getLabel().copyTo(label);
		_growCut.getStrength().copyTo( strength );

		itr_count++;
		if( itr_count % 10 == 0 ){
			//â¬éãâª
			for( int i=0; i<label.rows; i++ ){
				unsigned char* dst_line = dst.ptr(i);
				for( int h=0; h<label.cols; h++ ){
					unsigned char* dst_pix = dst_line + 3 * h;

					if( label.at<unsigned char>(i,h) == 1 ){
						dst_pix[0] = 0;
						dst_pix[1] = 0;
						dst_pix[2] = 255;
					}
					else if( label.at<unsigned char>(i,h) == 2 ){
						dst_pix[0] = 255;
						dst_pix[1] = 0;
						dst_pix[2] = 0;
					}
					else if( label.at<unsigned char>(i,h) == 3 ){
						dst_pix[0] = 0;
						dst_pix[1] = 255;
						dst_pix[2] = 0;
					}
					else {
						dst_line[h] = 128;
					}
				}
			}

			cout<< itr_count << " : " << converged <<endl;
			imshow( "Sgmentation result", dst );
			imshow( "strength", strength );
			int ikey = waitKey(33);
			char ckey = (char)ikey;
			if( ckey == 0x1b ){
				return;
			}
		}

		converged = _growCut.run_once();

	}while( converged > 0 );
}

int main( int argc, char** argv ){

	MouseEventContainer mparam;

	bool do_grow_cut;

	if( argc == 1 ){
		mparam.src_img = imread("neko.jpg");
		resize( mparam.src_img, mparam.src_img, Size( mparam.src_img.cols/2, mparam.src_img.rows/2));
		mparam.show_img = mparam.src_img.clone();

		mparam.label_img = mparam.src_img.clone();
		mparam.label_img.setTo( Scalar(0,0,0));

		mparam.wname = "GrowCut";
		string &wname = mparam.wname;
		namedWindow(wname);
		setMouseCallback( wname, mouse_func, (void*)&mparam );


		do_grow_cut = false;
		int ikey;
		do {

			imshow( wname, mparam.show_img );
			ikey = waitKey();
			char ckey = (char)ikey;

			if( ckey == 'c' ){
				mparam.mask_img.setTo( Scalar(0,0,0));
				genShowImage( mparam );
			}
			else if( ckey == 'g' ){
				do_grow_cut = true;
				break;
			}

		}while( ikey != 0x1b );

	}
	else if( argc == 3 ){
		Mat read_img = imread( argv[1] );
		if( read_img.data == NULL ){
			cerr<< "failed to read " << argv[1] <<endl;
			return 0;
		}
		Size resize_size( read_img.cols/4, read_img.rows/4 );
		resize( read_img, mparam.src_img, resize_size); 

		imshow( "src", mparam.src_img );
		cout<< "image type :" << mparam.src_img.type() <<endl;
		cout<< "correct image ?(type y/n)>" <<flush;
		if( (char)waitKey() == 'n' ){
			return 0;
		}
		destroyWindow( "src" );

		read_img = imread( argv[2] );
		if( read_img.data == NULL ){
			cerr<< "failed to read " << argv[2] <<endl;
			return 0;
		}
		resize( read_img, mparam.label_img, resize_size); 
		imshow( "label", mparam.label_img );
		cout<< "image type :" << mparam.label_img.type() <<endl;
		cout<< "correct image ?(type y/n)>" <<flush;
		if( (char)waitKey() == 'n' ){
			return 0;
		}
		destroyWindow( "label" );

		do_grow_cut = true;
	}


	if( do_grow_cut ){
		imshow( "original", mparam.src_img );
		Mat dst;
		growCut( mparam.src_img, mparam.label_img, dst);
		Mat crop_image;
		crop_image = mparam.src_img * 0.5 + dst * 0.5;
		imshow( "crop result", crop_image );
		cout<< "finish(type any key to exit)" <<endl;
		waitKey();
	}


	return 0;
}