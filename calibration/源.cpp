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
		if (k == 's')//��s����ͼƬ
		{
			imwrite(ImgName, img);
			ImgName.at(0)++;
			img.release();
		}
		else if (k == 27)//Esc��E
			break;
	}
	*/
	
	ifstream fin("calibdata.txt");             /* ��E�����ͼ���ļ���·�� */
	ofstream fout("caliberation_result.txt");  /* ���汁E�������ļ� */

	// ��ȡÿһ��ͼ�񣬴�����ȡ���ǵ㣬Ȼ��Խǵ���������ؾ�ȷ��
	int image_count = 0;  /* ͼ������ */
	Size image_size;      /* ͼ��ĳߴ�E*/
	Size board_size = Size(10, 7);             /* ��E�����ÿ�С��еĽǵ��� */
	vector<Point2f> image_points_buf;         /* ����ÿ��ͼ���ϼ�E⵽�Ľǵ�E*/
	vector<vector<Point2f>> image_points_seq; /* ���漁E⵽�����нǵ�E*/
	string filename;      // ͼƬÁE
	vector<string> filenames;

	while (getline(fin, filename))
	{
		++image_count;
		Mat imageInput = imread(filename);
		filenames.push_back(filename);

		// ��ȁE�һ��ͼƬʱ��ȡͼƬ��С
		if (image_count == 1)
		{
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
		}

		/* ��ȡ�ǵ�E*/
		if (0 == findChessboardCorners(imageInput, board_size, image_points_buf))
		{
			cout << "can not find chessboard corners!\n";  
			exit(1);
		}
		else
		{
			Mat view_gray;
			cvtColor(imageInput, view_gray, CV_RGB2GRAY);  // ת�Ҷ�ͼ

			/* �����ؾ�ȷ�� */
			// image_points_buf ��ʼ�Ľǵ�����E�����ͬʱ��Ϊ����������E��õ��䳁E
			// Size(5,5) �������ڴ�С
			// ��-1��-1�����û����ǁE
			// TermCriteria �ǵ�ĵ�E����̵���ֹ����, ����Ϊ��E������ͽǵ㾫�����ߵ����
			cornerSubPix(view_gray, image_points_buf, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

			image_points_seq.push_back(image_points_buf);  // ���������ؽǵ�E

			/* ��ͼ������ʾ�ǵ�λ�� */
			drawChessboardCorners(view_gray, board_size, image_points_buf, false); // ������ͼƬ�б�Eǽǵ�E

			//imshow("Camera Calibration", view_gray);       // ��ʾͼƬ

			//waitKey(500); //��ͣ     
		}
	}
	int CornerNum = board_size.width * board_size.height;  // ÿ��ͼƬ���ܵĽǵ���

	//-------------�������������E�------------------

	/*������ά��Ϣ*/
	Size square_size = Size(23, 23);         /* ʵ�ʲ����õ��ı�E�����ÿ�����̸�Ĵ�С */
	vector<vector<Point3f>> object_points;   /* ���汁E����Ͻǵ����ά����E*/

	/*�������*/
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));  /* ������ڲ�����ՁE*/
	vector<int> point_counts;   // ÿ��ͼ���нǵ������
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));       /* �������5������ϵ����k1,k2,p1,p2,k3 */
	vector<Mat> tvecsMat;      /* ÿ��ͼ���ƽ������ */
	vector<Mat> rvecsMat;      /* ÿ��ͼ�����ת���� */

	/* ��ʼ����E����Ͻǵ����ά����E*/
	int i, j, t;
	for (t = 0; t < image_count; t++)
	{
		vector<Point3f> tempPointSet;
		for (i = 0; i < board_size.height; i++)
		{
			for (j = 0; j < board_size.width; j++)
			{
				Point3f realPoint;
				/* ���豁E��������������E���z=0��ƽ���� */
				realPoint.x = i * square_size.width;
				realPoint.y = j * square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}

	/* ��ʼ��ÿ��ͼ���еĽǵ��������ٶ�ÿ��ͼ���ж����Կ���́E��ı�E���E*/
	for (i = 0; i < image_count; i++)
	{
		point_counts.push_back(board_size.width * board_size.height);
	}

	/* ��ʼ��E� */
	// object_points ��������E��еĽǵ����ά����E
	// image_points_seq ÿһ���ڽǵ��Ӧ��ͼ������E�E
	// image_size ͼ������سߴ��С
	// cameraMatrix ������ڲξ�ՁE
	// distCoeffs ���������ϵ��
	// rvecsMat �������ת����
	// tvecsMat �����λ������
	// 0 ��E�ʱ�����õ��㷨
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);

	//------------------------��E�́E�------------------------------------

	// -------------------�Ա�E������������------------------------------
	
	double total_err = 0.0;         /* ����ͼ���ƽ��������ܺ� */
	double err = 0.0;               /* ÿ��ͼ���ƽ����E*/
	vector<Point2f> image_points2moto;
	//vector<Point2f> image_points2x;
	Mat Rotation;
	Mat Rt;
	Mat pixel;
	Point2f pix;
	Mat world = Mat(4, 1, CV_64FC1, Scalar::all(0));
	cout << "Calibration error for each image��\n";

	for (i = 0; i < image_count; i++)
	{
		Mat test = imread(filenames[i]);
		vector<Point2f> image_points2;
		vector<Point3f> tempPointSet = object_points[i];
		/* ͨ���õ������������������Կռ����ά���������ͶӰ���㣬�õ��µ�ͶӰ��E*/
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
			
		
		//�����µ�ͶӰ��;ɵ�ͶӰ��֮�����E
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
		cout << "The NO." << i + 1 << " picture's Average reprojection error��" << err << " pixel" << endl;
		system("pause");
	}
	cout << " The whole average reprojection error��" << total_err / image_count << " pixel" << endl << endl;

	//-------------------------����́E�---------------------------------------------

	//-----------------------���涨��EṁE------------------------------------------ 
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));  /* ����ÿ��ͼ�����ת��ՁE*/
	cout << "Camera intrinsic parameters��" << endl;
	cout << cameraMatrix << endl << endl;
	cout << "Lens distortion parameters��\n";
	cout << distCoeffs << endl << endl << endl;
	/*
	for (int i = 0; i < image_count; i++)
	{
		cout << "��" << i + 1 << "��ͼ�����ת������" << endl;
		cout << tvecsMat[i] << endl;

		//����ת����ת��Ϊ���Ӧ����ת��ՁE
		Rodrigues(tvecsMat[i], rotation_matrix);
		cout << "��" << i + 1 << "��ͼ�����ת����" << endl;
		cout << rotation_matrix << endl;
		cout << "��" << i + 1 << "��ͼ���ƽ��������" << endl;
		cout << rvecsMat[i] << endl << endl;
	}
	*/
	cout << endl;

	//--------------------��E���������ʁE------------------------------

	//----------------------��ʾ����EṁE-------------------------------

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