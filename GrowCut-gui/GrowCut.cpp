#include "GrowCut.h"

#include <assert.h>

using namespace ys;

GrowCut::GrowCut(
	const cv::Mat& src,
	const cv::Mat& label )
	:_strength( src.size(), CV_64FC1 )
{
	assert( src.size() == label.size() );
	assert( label.type() == CV_8UC1 );

	src.copyTo( _image );
	//CV_8UC1‚ð‘z’è
	label.copyTo( _label );
	_next_label = label.clone();


	int rows = label.rows;
	int cols = label.cols;
	for( int i=0; i<rows; i++ ){
		uchar* line_label = _label.ptr(i);
		double* line_strength = (double*)_strength.ptr(i);

		for( int h=0; h<cols; h++ ){
			double label_val = line_label[h];

			if( label_val == 0 ){
				line_strength[h] = 0.0;
			}
			else {
				line_strength[h] = 1.0;
			}

		}
	}
	_next_strength = _strength.clone();
}

int GrowCut::run_once(){

	const int rows = _label.rows;
	const int cols = _label.cols;

	int converged = 0;

	for( int i=0; i<rows; i++ ){
		unsigned char* cline = (unsigned char*)_image.ptr(i);
		uchar* cline_label = _label.ptr(i);
		double* cline_strength = (double*)_strength.ptr(i);

		uchar* cline_next_label = _next_label.ptr(i);
		double* cline_next_strength = (double*)_next_strength.ptr(i);

		for( int h=0; h<cols; h++ ){
			unsigned char* cpix = cline + 3 * h;
			uchar* clabel = cline_label + h;
			double* cstrength = cline_strength + h;

			uchar* cnext_label = cline_next_label + h;
			double* cnext_strength = cline_next_strength + h;

			for( int neigh_y=-1; neigh_y<=1; neigh_y++ ){
				int i_neigh = i + neigh_y;

				if( i_neigh < 0 ||  rows <= i_neigh ) continue;

				unsigned char* nline = (unsigned char*)_image.ptr(i_neigh);
				uchar* nline_label = _label.ptr(i_neigh);
				double* nline_strength = (double*)_strength.ptr(i_neigh);

				for( int neigh_x=-1; neigh_x<=1; neigh_x++ ){
					int h_neigh = h + neigh_x;

					if( neigh_x == 0 && neigh_y == 0
						|| h_neigh < 0 || cols <= h_neigh) continue;

					unsigned char* npix = nline + 3 * (h_neigh);
					uchar* nlabel = nline_label + h_neigh;
					double* nstrength = nline_strength + h_neigh;

					double b = (double)npix[0] - cpix[0];
					double g = (double)npix[1] - cpix[1];
					double r = (double)npix[2] - cpix[2];

					double C = b*b + g*g + r*r; 
					double G = 1 - C / (3 * 255 * 255);

					/*
					if( G * strength.at<double>( i+neigh_y, h+neigh_x ) <= strength.at<double>( i, h )) continue;

					converged++;

					next_label.at<double>( i, h ) = label.at<double>( i+neigh_y, h+neigh_x );
					next_strength.at<double>( i, h ) = G * strength.at<double>( i+neigh_y, h+neigh_x );
					/*/
					if( G * *nstrength <= *cstrength ) continue;

					converged++;

					*cnext_label = *nlabel;
					*cnext_strength = G * *nstrength;
					//*/

				}
			}
		}
	}

	_next_label.copyTo( _label );
	_next_strength.copyTo( _strength );

	return converged;
}

void GrowCut::run(){
	int converged;
	do {
		converged = run_once();
	}while( converged > 0 );
}

const cv::Mat& GrowCut::getLabel(){
	return _label;
}

const cv::Mat& GrowCut::getStrength(){
	return _strength;
}