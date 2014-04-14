


#include <queue>
#include <list>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>

#include "tinydir.h"

#define PI 3.1415

#define HMAX 255
#define SMAX 255
#define VMAX 255

#define HBINS 16
#define SBINS 3
#define VBINS 3

const int HBIN_SIZE = HMAX / HBINS;
const int SBIN_SIZE = SMAX / SBINS;
const int VBIN_SIZE = VMAX / VBINS;


using namespace cv;
using namespace std;





// that function reads the filenames in the given directory and parses the images out of the sorted filenames
list<Mat> readImagesFromDirectory(string dirpath = "bilder") {
	
	list<Mat> imagelist;
	list<string> filenames;	
	// create directory object and open it
	tinydir_dir dir;
	tinydir_open(&dir, dirpath.c_str());

	while (dir.has_next)
	{
		// create file
		tinydir_file file;
		tinydir_readfile(&dir, &file);
		
		
		// if file isnt a directory and if file isnt the DICOMDIR file
		if (!file.is_dir)
		{
			// parse 
			filenames.push_back(file.name);
		}
		
		// read next file
		tinydir_next(&dir);
	}
	
	// close directory object
	tinydir_close(&dir);
	
	// sort filenames
	filenames.sort();

	list<string>::iterator iter;
	int i = 0;
	for(iter = filenames.begin(); iter != filenames.end(); iter++)
	{
		// create image from file
		Mat img = imread(dirpath+"/"+(*iter), -1);
		if(img.data)
		{
			imagelist.push_back(img);
			/*ostringstream os;
			os << "original " << i << ".png";
			namedWindow(os.str(), CV_WINDOW_AUTOSIZE);
			imshow(os.str(), img);*/
		}
		i++;
	}

	return imagelist;
}





// old way, trying to understand opncv calcHist quark
Mat getCVHist(Mat img) {
	Mat hist;
	Mat hsv;
	cvtColor(img, hsv, CV_BGR2HSV);
	int channels[] = {0, 1, 2};
	int sbins = 3;
	int hbins = 16;
	int vbins = 3;
	int histSizes[] = {hbins, sbins, vbins};
	float hrange[] = {0, 180};
	float srange[] = {0, 256};
	float vrange[] = {0, 256};
	const float* ranges[] = {hrange, srange, vrange};
	calcHist(&hsv, 1, channels, Mat(), hist, 3, histSizes, ranges);
	return hist;
}

// new way: recommended by professor
// simply run over the image and put the pixel into that bin, the pixel belongs to
double* getHist(Mat img) {
	// create hist, initialized with zeros
	double *hist = (double *) calloc(sizeof(double), HBINS * SBINS * VBINS);
	
	// compute number of pixels of the given image
	double size = img.rows * img.cols;

	// run over the image
	for(int r = 0; r < img.rows; r++)
	{
		for(int c = 0; c < img.cols; c++)
		{
			// obtain the HSV value of pixel
			double h = 0, s = 0, v = 0;
			int hbin = 0, sbin = 0, vbin = 0;
			Vec3b pixel = img.at<Vec3b>(r, c);
			h = pixel[0];
			s = pixel[1];
			v = pixel[2];

			// compute bin, that pixel belongs to
			hbin = (h / HBIN_SIZE);
			sbin = (s / SBIN_SIZE);
			vbin = (v / VBIN_SIZE);
			// increase value of that bin by (1 / size), to not dismiss large images
			hist[(hbin * (SBINS * VBINS)) + (sbin * VBINS) + vbin] += (1.0 / size);
			//cout << hist[10] << endl;
		}
	}
	return hist;
}


// simply builds the hist of each image in the given list
list<double*> buildHists(list<Mat> images) {
	list<double*> res;
	list<Mat>::iterator iter;
	for(iter = images.begin(); iter != images.end(); iter++)
	{
		res.push_back(getHist(*iter));
	}
	return res;
}







double l1dist(double *h1, double *h2) {
	/*
	cout << "nrows = " << h1.rows << endl;
	cout << "ncols = " << h1.cols << endl;
	cout << "type = " << h1.type() << endl;
	cout << "channels = " << h1.channels() << endl;
	cout << "dims = " << h1.dims << endl;
	
	
	double diff = 0.0;
	double hdiff = 0.0;
	double sdiff = 0.0;
	double vdiff = 0.0;

	for(int h = 0; h < 16; h++)
	{
		cout << "(" << h << ", 0, 0) = " << h1.at<double>(h, 0, 0) << endl;
		hdiff += fabs(h1.at<double>(h, 0, 0) - h2.at<double>(h,0,0));
	}

	
	for(int s = 0; s < 3; s++)
	{
		cout << "(0, " << s << ", 0) = " << h1.at<double>(0, s, 0) << endl;
		sdiff += fabs(h1.at<double>(0, s, 0) - h2.at<double>(0, s, 0));
	}

	for(int v = 0; v < 3; v++)
	{
		cout << "(0, 0, " << v << ") = " << h1.at<double>(0, 0, v) << endl;
		vdiff += fabs(h1.at<double>(0, 0, v) - h2.at<double>(0, 0, v));
	}

	diff = hdiff + sdiff + vdiff;
	*/


	// with the new hist function, we just have to add the difference of each bin to our diff and return it
	double diff = 0.0;
	for(int i = 0; i < (HBINS * SBINS * VBINS); i++)
	{
		diff += fabs(h1[i] - h2[i]);
	}


	return diff;
}


// converts one image into an HSV-image
Mat toHSV(Mat image) {
	Mat hsv;
	cvtColor(image, hsv, CV_BGR2HSV);
	return hsv;
}


// converts a list of images to a list of HSV-images
list<Mat> toHSV(list<Mat> images) {
	list<Mat>::iterator iter;
	list<Mat> res;
	for(iter = images.begin(); iter != images.end(); iter++)
	{	
		res.push_back(toHSV(*iter));
	}
	return res;
}




// needed for priority-queue
class mycomparison
{
	bool reverse;
	public:
		mycomparison(const bool& revparam = false)
		{
			reverse = revparam;
		}

		bool operator() (const pair<double, Mat>& lhs, const pair<double, Mat>& rhs) const
		{
			if (reverse)
				return (lhs.first > rhs.first);
			else
				return (lhs.first < rhs.first);
		}
};




void clearMemory(list<double *> hists) {
	list<double *>::iterator iter;
	
	for(iter = hists.begin(); iter != hists.end(); iter++)
	{
		free(*iter);
	}

}





int main( int argc, char** argv )
{

	string filename = "bilder/220px-Bananas.jpg";


	for(int i = 0; i < argc; i++)
	{
		if(strcmp(argv[i], "-ref") == 0 && i+1 < argc)
		{
			filename = argv[++i];
		}
	}
	

	
	// load reference image
	Mat refImage = imread(filename, -1);

	if(!refImage.data)
	{
		cout << "fail\n";
		return -1;
	}
	
	// convert reference image to hsv
	Mat refImagehsv = toHSV(refImage);
	// build reference hist
	double *refHist = getHist(refImagehsv);
	
	
	// show the orinal image
	namedWindow("original image", CV_WINDOW_AUTOSIZE);
	imshow("original image", refImage);



	// read images from directory
	list<Mat> images = readImagesFromDirectory();

	// convert images to HSV
	list<Mat> hsvImages = toHSV(images);

	// build hists of hsv-images
	list<double*> hists = buildHists(hsvImages);


	list<double*>::iterator iter = hists.begin();
	list<Mat>::iterator iterImg = images.begin();	
	priority_queue<pair <double, Mat>, vector<pair<double, Mat> >, mycomparison > pq(mycomparison(true));

	
	// compute difference and put the images with the difference into a priority-queue
	for(iter = hists.begin(), iterImg = images.begin(); iter != hists.end(); iter++, iterImg++)
	{
	
		double diff = l1dist(refHist, (*iter));
		pq.push(pair<double, Mat>(diff, (*iterImg)));
	}
	

	// show the 10 hottest
	for(int i = 1; i <= 10; i++)
	{
		pair<double, Mat> p = pq.top();
		pq.pop();
		ostringstream os;
		os << "rank " << i;
		namedWindow(os.str(), CV_WINDOW_AUTOSIZE);
		imshow(os.str(), p.second);
		cout << os.str() << " with diff = " << p.first << endl; 
	}
	

	waitKey(0);
	



	// clean up
	clearMemory(hists);
	free(refHist);


	return 0;
}






























