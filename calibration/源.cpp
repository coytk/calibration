#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <fstream>
#include<opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


void main()
{
	/*
	Mat img;
	int k;
	string ImgName = "1.jpg";
	VideoCapture cap(0);
	if (!cap.isOpened())
		exit(1);
	while (1) {
		cap >> img;
		GaussianBlur(img, img, Size(3, 3), 0);
		imshow("1", img);
		k = waitKey(30);
		if (k == 's')//°´s±£´æÍ¼Æ¬
		{
			imwrite(ImgName, img);
			ImgName.at(0)++;
			img.release();
		}
		else if (k == 27)//Esc¼E
			break;
	}
	*/
	
	ifstream fin("calibdata.txt");             /* ±E¨ËùÓÃÍ¼ÏñÎÄ¼şµÄÂ·¾¶ */
	ofstream fout("caliberation_result.txt");  /* ±£´æ±E¨½á¹ûµÄÎÄ¼ş */

	// ¶ÁÈ¡Ã¿Ò»·ùÍ¼Ïñ£¬´ÓÖĞÌáÈ¡³ö½Çµã£¬È»ºó¶Ô½Çµã½øĞĞÑÇÏñËØ¾«È·»¯
	int image_count = 0;  /* Í¼ÏñÊıÁ¿ */
	Size image_size;      /* Í¼ÏñµÄ³ß´E*/
	Size board_size = Size(10, 7);             /* ±E¨°åÉÏÃ¿ĞĞ¡¢ÁĞµÄ½ÇµãÊı */
	vector<Point2f> image_points_buf;         /* »º´æÃ¿·ùÍ¼ÏñÉÏ¼Eâµ½µÄ½ÇµE*/
	vector<vector<Point2f>> image_points_seq; /* ±£´æ¼Eâµ½µÄËùÓĞ½ÇµE*/
	string filename;      // Í¼Æ¬ÃE
	vector<string> filenames;

	while (getline(fin, filename))
	{
		++image_count;
		Mat imageInput = imread(filename);
		filenames.push_back(filename);

		// ¶ÁÈEÚÒ»ÕÅÍ¼Æ¬Ê±»ñÈ¡Í¼Æ¬´óĞ¡
		if (image_count == 1)
		{
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
		}

		/* ÌáÈ¡½ÇµE*/
		if (0 == findChessboardCorners(imageInput, board_size, image_points_buf))
		{
			cout << "can not find chessboard corners!\n";  
			exit(1);
		}
		else
		{
			Mat view_gray;
			cvtColor(imageInput, view_gray, CV_RGB2GRAY);  // ×ª»Ò¶ÈÍ¼

			/* ÑÇÏñËØ¾«È·»¯ */
			// image_points_buf ³õÊ¼µÄ½Çµã×ø±EòÁ¿£¬Í¬Ê±×÷ÎªÑÇÏñËØ×ø±E»ÖÃµÄÊä³E
			// Size(5,5) ËÑË÷´°¿Ú´óĞ¡
			// £¨-1£¬-1£©±úæ¾Ã»ÓĞËÀÇE
			// TermCriteria ½ÇµãµÄµEú¹ı³ÌµÄÖÕÖ¹Ìõ¼ş, ¿ÉÒÔÎªµEú´ÎÊıºÍ½Çµã¾«¶ÈÁ½ÕßµÄ×éºÏ
			cornerSubPix(view_gray, image_points_buf, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

			image_points_seq.push_back(image_points_buf);  // ±£´æÑÇÏñËØ½ÇµE

			/* ÔÚÍ¼ÏñÉÏÏÔÊ¾½ÇµãÎ»ÖÃ */
			drawChessboardCorners(view_gray, board_size, image_points_buf, false); // ÓÃÓÚÔÚÍ¼Æ¬ÖĞ±EÇ½ÇµE

			//imshow("Camera Calibration", view_gray);       // ÏÔÊ¾Í¼Æ¬

			//waitKey(500); //ÔİÍ£     
		}
	}
	int CornerNum = board_size.width * board_size.height;  // Ã¿ÕÅÍ¼Æ¬ÉÏ×ÜµÄ½ÇµãÊı

	//-------------ÒÔÏÂÊÇÉãÏñ»ú±E¨------------------

	/*ÆåÅÌÈıÎ¬ĞÅÏ¢*/
	Size square_size = Size(23, 23);         /* Êµ¼Ê²âÁ¿µÃµ½µÄ±E¨°åÉÏÃ¿¸öÆåÅÌ¸ñµÄ´óĞ¡ */
	vector<vector<Point3f>> object_points;   /* ±£´æ±E¨°åÉÏ½ÇµãµÄÈıÎ¬×ø±E*/

	/*ÄÚÍâ²ÎÊı*/
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));  /* ÉãÏñ»úÄÚ²ÎÊı¾ØÕE*/
	vector<int> point_counts;   // Ã¿·ùÍ¼ÏñÖĞ½ÇµãµÄÊıÁ¿
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));       /* ÉãÏñ»úµÄ5¸ö»û±äÏµÊı£ºk1,k2,p1,p2,k3 */
	vector<Mat> tvecsMat;      /* Ã¿·ùÍ¼ÏñµÄÆ½ÒÆÏòÁ¿ */
	vector<Mat> rvecsMat;      /* Ã¿·ùÍ¼ÏñµÄĞı×ªÏòÁ¿ */

	/* ³õÊ¼»¯±E¨°åÉÏ½ÇµãµÄÈıÎ¬×ø±E*/
	int i, j, t;
	for (t = 0; t < image_count; t++)
	{
		vector<Point3f> tempPointSet;
		for (i = 0; i < board_size.height; i++)
		{
			for (j = 0; j < board_size.width; j++)
			{
				Point3f realPoint;
				/* ¼ÙÉè±E¨°å·ÅÔÚÊÀ½ç×ø±EµÖĞz=0µÄÆ½ÃæÉÏ */
				realPoint.x = i * square_size.width;
				realPoint.y = j * square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}

	/* ³õÊ¼»¯Ã¿·ùÍ¼ÏñÖĞµÄ½ÇµãÊıÁ¿£¬¼Ù¶¨Ã¿·ùÍ¼ÏñÖĞ¶¼¿ÉÒÔ¿´µ½ÍEûµÄ±E¨°E*/
	for (i = 0; i < image_count; i++)
	{
		point_counts.push_back(board_size.width * board_size.height);
	}

	/* ¿ªÊ¼±E¨ */
	// object_points ÊÀ½ç×ø±EµÖĞµÄ½ÇµãµÄÈıÎ¬×ø±E
	// image_points_seq Ã¿Ò»¸öÄÚ½Çµã¶ÔÓ¦µÄÍ¼Ïñ×ø±EE
	// image_size Í¼ÏñµÄÏñËØ³ß´ç´óĞ¡
	// cameraMatrix Êä³ö£¬ÄÚ²Î¾ØÕE
	// distCoeffs Êä³ö£¬»û±äÏµÊı
	// rvecsMat Êä³ö£¬Ğı×ªÏòÁ¿
	// tvecsMat Êä³ö£¬Î»ÒÆÏòÁ¿
	// 0 ±E¨Ê±Ëù²ÉÓÃµÄËã·¨
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);

	//------------------------±E¨ÍEÉ------------------------------------

	// -------------------¶Ô±E¨½á¹û½øĞĞÆÀ¼Û------------------------------
	
	double total_err = 0.0;         /* ËùÓĞÍ¼ÏñµÄÆ½¾ùÎó²ûÑÄ×ÜºÍ */
	double err = 0.0;               /* Ã¿·ùÍ¼ÏñµÄÆ½¾ùÎó²E*/
	vector<Point2f> image_points2moto;
	//vector<Point2f> image_points2x;
	Mat Rotation;
	Mat Rt;
	Mat pixel;
	Point2f pix;
	Mat world = Mat(4, 1, CV_64FC1, Scalar::all(0));
	cout << "Calibration error for each image£º\n";

	for (i = 0; i < image_count; i++)
	{
		Mat test = imread(filenames[i]);
		vector<Point2f> image_points2;
		vector<Point3f> tempPointSet = object_points[i];
		/* Í¨¹ıµÃµ½µÄÉãÏñ»úÄÚÍâ²ÎÊı£¬¶Ô¿Õ¼äµÄÈıÎ¬µã½øĞĞÖØĞÂÍ¶Ó°¼ÆËã£¬µÃµ½ĞÂµÄÍ¶Ó°µE*/
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2moto);
		cout << image_points2moto << endl;
		
		Rodrigues(rvecsMat[i], Rotation);
		hconcat(Rotation,tvecsMat[i],Rt);
		for (int n = 0; n < CornerNum; n++) {
			world.at<double>(0, 0) = tempPointSet[n].x;
			world.at<double>(1, 0) = tempPointSet[n].y;
			world.at<double>(2, 0) = tempPointSet[n].z;
			world.at<double>(3, 0) = 1;			
			pixel = cameraMatrix * Rt * world;
			pix.x = pixel.at<double>(0, 0) / pixel.at<double>(2, 0);
			pix.y = pixel.at<double>(1, 0) / pixel.at<double>(2, 0);
			image_points2.push_back(pix);
		}	
		//VideoCapture cap(0);
		//Mat frame;
		//cap.read(frame);
		for (int a = 0; a < CornerNum; a++) {
			circle(test, image_points2[a], 3, Scalar(255, 0, 0), -1);
			circle(test, image_points2moto[a], 2, Scalar(0, 0, 255), -1);
		}
			imshow("Camera Calibration", test);
			waitKey(300);
			
		
		//¼ÆËãĞÂµÄÍ¶Ó°µãºÍ¾ÉµÄÍ¶Ó°µãÖ®¼äµÄÎó²E
		vector<Point2f> tempImagePoint = image_points_seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= point_counts[i];
		cout << "The NO." << i + 1 << " picture's Average reprojection error£º" << err << " pixel" << endl;
		system("pause");
	}
	cout << " The whole average reprojection error£º" << total_err / image_count << " pixel" << endl << endl;

	//-------------------------ÆÀ¼ÛÍEÉ---------------------------------------------

	//-----------------------±£´æ¶¨±Eá¹E------------------------------------------ 
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));  /* ±£´æÃ¿·ùÍ¼ÏñµÄĞı×ª¾ØÕE*/
	cout << "Camera intrinsic parameters£º" << endl;
	cout << cameraMatrix << endl << endl;
	cout << "Lens distortion parameters£º\n";
	cout << distCoeffs << endl << endl << endl;
	/*
	for (int i = 0; i < image_count; i++)
	{
		cout << "µÚ" << i + 1 << "·ùÍ¼ÏñµÄĞı×ªÏòÁ¿£º" << endl;
		cout << tvecsMat[i] << endl;

		//½«Ğı×ªÏòÁ¿×ª»»ÎªÏà¶ÔÓ¦µÄĞı×ª¾ØÕE
		Rodrigues(tvecsMat[i], rotation_matrix);
		cout << "µÚ" << i + 1 << "·ùÍ¼ÏñµÄĞı×ª¾ØÕó£º" << endl;
		cout << rotation_matrix << endl;
		cout << "µÚ" << i + 1 << "·ùÍ¼ÏñµÄÆ½ÒÆÏòÁ¿£º" << endl;
		cout << rvecsMat[i] << endl << endl;
	}
	*/
	cout << endl;

	//--------------------±E¨½á¹û±£´æ½áÊE------------------------------

	//----------------------ÏÔÊ¾¶¨±Eá¹E-------------------------------

	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	string imageFileName;
	std::stringstream StrStm;
	for (int i = 0; i != image_count; i++)
	{
		initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);
		Mat imageSource = imread(filenames[i]);
		Mat newimage = imageSource.clone();
		remap(imageSource, newimage, mapx, mapy, INTER_LINEAR);
		StrStm.clear();
		imageFileName.clear();
		StrStm << i + 1;
		StrStm >> imageFileName;
		imageFileName += "_d.jpg";
		imwrite(imageFileName, newimage);
	}
	waitKey();
	fin.close();
	fout.close();
	return;
}