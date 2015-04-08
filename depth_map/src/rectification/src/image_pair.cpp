#include "image_pair.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "daisy/daisy.h"

namespace rect
{

	ImagePair::ImagePair()
	{
		Clear();
	}

	ImagePair::ImagePair( std::string yaml_flename )
	{
		Clear();

		if (ImportFromYAML(yaml_flename))
		{
			std::cout << "load camera from "<<yaml_flename<<" failed."<<std::endl;
			_left_image = Camera();
			_right_image = Camera();
		}
	}

	ImagePair::ImagePair( const Camera & left, const Camera & right )
	{
		Clear();

		_left_image = left;
		_right_image = right;
	}

	ImagePair::~ImagePair()
	{
	}

	int ImagePair::ImportFromYAML( std::string yaml_filename )
	{
		int ret = -1;

		do 
		{
			_left_image = Camera();
			_right_image = Camera();

			cv::FileStorage fs(yaml_filename, cv::FileStorage::READ);
			if(!fs.isOpened()) break;


			Camera left, right;

			fs["filename_l"] >> left._filename;
			fs["M_l"] >> left._CamMatrix;
			fs["D_l"] >> left._Distortion;
			fs["R_l"] >> left._R;
			fs["C_l"] >> left._C;
			if (_load_image_in_advanced)
			{
				left._image = cv::imread(left._filename,
					_load_image_in_gray_scale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED);
				if (left._image.empty()) break;
			}

			fs["filename_r"] >> right._filename;
			fs["M_r"] >> right._CamMatrix;
			fs["D_r"] >> right._Distortion;
			fs["R_r"] >> right._R;
			fs["C_r"] >> right._C;
			if (_load_image_in_advanced)
			{
				right._image = cv::imread(right._filename,
					_load_image_in_gray_scale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED);
				if (right._image.empty()) break;
			}


			// modified R due to the direction of R matrix by bundler
			cv::Mat tmp = left._R.row(0);
			tmp *= -1.;
			tmp.copyTo(left._R.row(0));
			tmp = left._R.row(2);
			tmp *= -1.;
			tmp.copyTo(left._R.row(2));

			tmp = right._R.row(0);
			tmp *= -1.;
			tmp.copyTo(right._R.row(0));
			tmp = right._R.row(2);
			tmp *= -1.;
			tmp.copyTo(right._R.row(2));


			_left_image = left;
			_right_image = right;

			
			if (image_pair_validation()) break;


			ret = 0;
		} while (0);
error0:

		return ret;
	}

	void ImagePair::Clear()
	{
		_left_image = Camera();
		_right_image = Camera();

		_load_image_in_gray_scale = false;
		_load_image_in_advanced = false;
	}

	int ImagePair::SaveRectifiedPair( std::string file_prefix )
	{
		int ret = -1;

		do 
		{
			cv::Mat left, right;
			if (_load_image_in_advanced)
			{
				left = _left_image._image;
				right = _right_image._image;
			} else {
				left = cv::imread(_left_image._filename, 
					_load_image_in_gray_scale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED);
				right = cv::imread(_right_image._filename, 
					_load_image_in_gray_scale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED);
			}
			if (left.empty() || right.empty()) break;


			cv::Mat R, T, R1, R2, P1, P2, Q;
			cv::Mat rectified_left, rectified_right;
			cv::Mat map11, map12, map21, map22;
			
			R = _right_image._R * _left_image._R.inv();
			T = _right_image._R * (_left_image._C-_right_image._C);
			cv::stereoRectify( _left_image._CamMatrix, _left_image._Distortion,
				_right_image._CamMatrix, _right_image._Distortion,
				left.size(), R, T, R1, R2, P1, P2, Q);

			initUndistortRectifyMap(_left_image._CamMatrix, _left_image._Distortion,
				R1, P1, left.size(), CV_32FC1, map11, map12);
			initUndistortRectifyMap(_right_image._CamMatrix, _right_image._Distortion,
				R2, P2, right.size(), CV_32FC1, map21, map22);
			remap(left, rectified_left, map11, map12, cv::INTER_LINEAR);
			remap(right, rectified_right, map21, map22, cv::INTER_LINEAR);

			imwrite(file_prefix + "_left.jpg", rectified_left);
			imwrite(file_prefix + "_right.jpg", rectified_right);


			double m[3] = {1200,800,1};
			cv::Mat xy(3,1,R1.type(), m);
			cv::Mat K_n(3,3,R1.type());
			cv::Mat K(3,3,R1.type());
			_left_image._CamMatrix.convertTo(K, K.type());
			P1.col(0).copyTo(K_n.col(0));
			P1.col(1).copyTo(K_n.col(1));
			P1.col(2).copyTo(K_n.col(2));

// 			xy = K_n.inv() * xy;
// 			xy = R1.inv() * xy;
// 			xy /= xy.at<double>(2);
// 			xy = K * xy;

			xy = K.inv() * xy;
			xy = R1 * xy;
			xy /= xy.at<double>(2);
			xy = K_n * xy;

			std::cout<<K_n << std::endl;
			std::cout<<K << std::endl;
			std::cout<<xy<<std::endl;


			ret = 0;
		} while (0);
error0:

		return ret;
	}

	int ImagePair::image_pair_validation()
	{
		if ( (_load_image_in_advanced && 
			(_left_image._image.empty() || _right_image._image.empty())) ||
			_left_image._C.rows != 3 || _left_image._C.cols !=1 || 
			_right_image._C.rows != 3 || _right_image._C.cols !=1 ||
			_left_image._R.rows != 3 || _left_image._R.cols !=3 || 
			_right_image._R.rows != 3 || _right_image._R.cols !=3 ||
			_left_image._CamMatrix.rows != 3 || _left_image._CamMatrix.cols !=3 || 
			_right_image._CamMatrix.rows != 3 || _right_image._CamMatrix.cols !=3) return -1;
		else return 0;
	}

	int ImagePair::SimpleDaisyDense( bool save_disparity /*= false*/ )
	{
		int ret = -1;

		do 
		{
			Eigen::Matrix3d Rl, Rr, Kl, Kr, Kl_n, Kr_n;
			std::string rect_l = "_left.jpg";
			std::string rect_r = "_right.jpg";
			std::string pc_filename = "pc.xyz";
			bool is_vertical_rectified = false;
			double daisy_match_threshold = 0.8;
			int best_match_id_dist_threshold = 2;

			int rad   = 15;
			int radq  =  3;
			int thq   =  8;
			int histq =  8;
			double pi = 3.14159265358979323846;
			double sigma_2 = 0.2;
			double Z = 1. / sqrt(2. * pi * sigma_2);
			double focal_l = _left_image._CamMatrix.at<double>(0,0);
			double baseline_len = sqrt(norm(_left_image._C - _right_image._C));

			// recitify image pair
			{
				cv::Mat left, right;
				if (_load_image_in_advanced)
				{
					left = _left_image._image;
					right = _right_image._image;
				} else {
					left = cv::imread(_left_image._filename, 
						_load_image_in_gray_scale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED);
					right = cv::imread(_right_image._filename, 
						_load_image_in_gray_scale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED);
				}
				if (left.empty() || right.empty()) break;


				cv::Mat R, T, R1, R2, P1, P2, Q;
				cv::Mat rectified_left, rectified_right;
				cv::Mat map11, map12, map21, map22;

				R = _right_image._R * _left_image._R.inv();
				T = _right_image._R * (_left_image._C-_right_image._C);
				cv::stereoRectify( _left_image._CamMatrix, _left_image._Distortion,
					_right_image._CamMatrix, _right_image._Distortion,
					left.size(), R, T, R1, R2, P1, P2, Q);

				initUndistortRectifyMap(_left_image._CamMatrix, _left_image._Distortion,
					R1, P1, left.size(), CV_32FC1, map11, map12);
				initUndistortRectifyMap(_right_image._CamMatrix, _right_image._Distortion,
					R2, P2, right.size(), CV_32FC1, map21, map22);
				remap(left, rectified_left, map11, map12, cv::INTER_LINEAR);
				remap(right, rectified_right, map21, map22, cv::INTER_LINEAR);

				imwrite(rect_l, rectified_left);
				imwrite(rect_r, rectified_right);

				
				if (abs(P2.at<double>(1,3)) > abs(P2.at<double>(0,3)))
					is_vertical_rectified = true;

				
				Kl = Eigen::Map<Eigen::MatrixXd>(
					reinterpret_cast<double*>(_left_image._CamMatrix.data),
					_left_image._CamMatrix.rows,
					_left_image._CamMatrix.cols).transpose();
				Kr = Eigen::Map<Eigen::MatrixXd>(
					reinterpret_cast<double*>(_right_image._CamMatrix.data),
					_right_image._CamMatrix.rows,
					_right_image._CamMatrix.cols).transpose();
				Rl = Eigen::Map<Eigen::MatrixXd>(
					reinterpret_cast<double*>(R1.data),
					R1.rows, R1.cols).transpose();
				Rr = Eigen::Map<Eigen::MatrixXd>(
					reinterpret_cast<double*>(R2.data),
					R2.rows, R2.cols).transpose();

				cv::Mat K_n(3,3,R1.type());
				P1.col(0).copyTo(K_n.col(0));
				P1.col(1).copyTo(K_n.col(1));
				P1.col(2).copyTo(K_n.col(2));
				Kl_n = Eigen::Map<Eigen::MatrixXd>(
					reinterpret_cast<double*>(K_n.data),
					K_n.rows, K_n.cols).transpose();
				P2.col(0).copyTo(K_n.col(0));
				P2.col(1).copyTo(K_n.col(1));
				P2.col(2).copyTo(K_n.col(2));
				Kr_n = Eigen::Map<Eigen::MatrixXd>(
					reinterpret_cast<double*>(K_n.data),
					K_n.rows, K_n.cols).transpose();
			}


			// generate dense point cloud
			{
				int w,h;
				uchar* im_l = NULL;
				uchar* im_r = NULL;
				daisy * desc_l = new daisy();
				daisy * desc_r = new daisy();


				bool dense_match = false;
				do 
				{
					if (kutility::load_gray_image (rect_l, im_l, h, w))
						break;
					desc_l->set_image(im_l,h,w);
					desc_l->verbose( /*verbose_level*/0 ); // 0,1,2,3 -> how much output do you want while running
					desc_l->set_parameters(rad, radq, thq, histq); // default values are 15,3,8,8
					desc_l->initialize_single_descriptor_mode();
					desc_l->compute_descriptors(); // precompute all the descriptors (NOT NORMALIZED!)
					// the descriptors are not normalized yet
					desc_l->normalize_descriptors();

					if (kutility::load_gray_image (rect_r, im_r, h, w))
						break;
					desc_r->set_image(im_r,h,w);
					desc_r->verbose( /*verbose_level*/0 ); // 0,1,2,3 -> how much output do you want while running
					desc_r->set_parameters(rad, radq, thq, histq); // default values are 15,3,8,8
					desc_r->initialize_single_descriptor_mode();
					desc_r->compute_descriptors(); // precompute all the descriptors (NOT NORMALIZED!)
					// the descriptors are not normalized yet
					desc_r->normalize_descriptors();


					std::ofstream pc_file(pc_filename);
					if (!pc_file.is_open()) break;
					pc_file << std::setprecision(15);

					for (int i_yl = 0; i_yl < h; ++i_yl)
					{
						for (int i_xl = 0; i_xl < w; ++i_xl)
						{
							Eigen::Vector3d xy_l(i_xl, i_yl, 1);
							Eigen::Vector3d valid_xy = xy_l;
							valid_xy = Rl.inverse() * Kl_n.inverse() * valid_xy;
							valid_xy /= valid_xy(2);
							valid_xy = Kl * valid_xy;

							if (valid_xy(0) < 0. || valid_xy(0) >= w ||
								valid_xy(1) < 0. || valid_xy(1) >= h)
							{
								continue;
							}

							int num_desc = desc_l->descriptor_size();
							double best_match = std::numeric_limits<double>::max();
							int best_match_id = -1;
							double second_best_match = std::numeric_limits<double>::max();
							int sec_best_match_id = -1;

							float* thor_l = NULL;
							desc_l->get_descriptor(i_yl,i_xl,thor_l);
							int dim_num = is_vertical_rectified ? h : w;

#ifdef USE_OPENMP
							omp_lock_t writelock;
							omp_init_lock(&writelock);
#pragma omp parallel for							
#endif // USE_OPENMP
							for (int i_dim = 0; i_dim < dim_num; ++i_dim)
							{
								int xr = is_vertical_rectified ? xy_l(0) : i_dim;
								int yr = is_vertical_rectified ? i_dim : xy_l(1);

								Eigen::Vector3d xy_r(xr, yr, 1);
								xy_r = Rr.inverse() * Kr_n.inverse() * xy_r;
								xy_r /= xy_r(2);
								xy_r = Kr * xy_r;

								if (xy_r(0) < 0. || xy_r(0) >= w ||
									xy_r(1) < 0. || xy_r(1) >= h)
								{
									continue;
								}


								float* thor_r = NULL;
								desc_r->get_descriptor(yr,xr,thor_r);

								double diff = 0.;
								for (int i_d = 0; i_d < num_desc; ++i_d)
								{
									diff += (thor_l[i_d] - thor_r[i_d]) * (thor_l[i_d] - thor_r[i_d]);
								}

								if (diff < best_match)
								{
#ifdef USE_OPENMP
									omp_set_lock(&writelock);			
#endif // USE_OPENMP

									second_best_match = best_match;
									sec_best_match_id = best_match_id;

									best_match = diff;
									best_match_id = i_dim;

#ifdef USE_OPENMP
									omp_unset_lock(&writelock);			
#endif // USE_OPENMP
								} else if (diff < second_best_match) {
#ifdef USE_OPENMP
									omp_set_lock(&writelock);			
#endif // USE_OPENMP
									second_best_match = diff;
									sec_best_match_id = i_dim;
#ifdef USE_OPENMP
									omp_unset_lock(&writelock);			
#endif // USE_OPENMP
								}
							}
#ifdef USE_OPENMP
							omp_destroy_lock(&writelock);	
#endif // USE_OPENMP

							// disparity
							if (best_match < daisy_match_threshold)
							{
								double ratio = best_match / second_best_match;
								if (ratio <= 0.8 ||
									abs(best_match_id - sec_best_match_id) < best_match_id_dist_threshold)
								{
									// test
									if (best_match_id >= xy_l(1)) continue;

									double disparity = is_vertical_rectified ?
										best_match_id - xy_l(1) : best_match_id - xy_l(0);

									disparity = abs(disparity);
									if (disparity == 0.) continue;

									double depth = focal_l * baseline_len / disparity;

									Eigen::Vector3d img_world_pt = depth * Kl.inverse() * xy_l;
									if (pc_file.good())
									{
										pc_file << img_world_pt(0) << " " <<
											img_world_pt(1) << " " <<
											img_world_pt(2) << std::endl;
									}
								}
							}

						}
					}

					dense_match = true;
				} while (0);
				
				deallocate(im_l);
				deallocate(im_r);
				if (desc_l) delete desc_l;
				if (desc_r) delete desc_r;

				if (!dense_match) break;
			}
			
			ret = 0;
		} while (0);
error0:

		return ret;
	}

}