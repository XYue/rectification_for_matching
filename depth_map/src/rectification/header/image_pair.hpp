#pragma once

#include <opencv2/core/core.hpp>

namespace rect
{
	class ImagePair
	{
		// member structures and types
	public:
		typedef struct Camera
		{
			std::string _filename;
			cv::Mat		_image;
			cv::Mat		_R;
			cv::Mat		_C;
			cv::Mat		_CamMatrix;
			cv::Mat		_Distortion;
		} Camera;


		// member functions
	public:
		ImagePair();
		ImagePair(std::string yaml_flename);
		ImagePair(const Camera & left, const Camera & right);
		~ImagePair();

		void Clear();

		// import camera thing from yaml file
		int ImportFromYAML(std::string yaml_filename);

		// write down rectified image pair	
		int SaveRectifiedPair(std::string file_prefix);

		// simple dense method
		int SimpleDaisyDense(bool save_disparity = false);

	protected:
		// validates the parameters of image pair
		// 0 successful; otherwise, failed.
		int image_pair_validation();
		
		// compare the current value and the best two
		// -1 smaller than both, 0 second best, 1 best
		inline int compare(double value, double best_value, double second_best_value);

		// compare the current value and the best two and reset the value if needed
		inline void compare(double value, int id,
			double & best_value, int & best_id,
			double & second_best_value, int & second_best_id);

		inline void compare(double value, double pos,
			double & best_value, double & best_pos,
			double & second_best_value, double & second_best_pos);

		// member variables
	protected:
		// load image into cv::Mat in advanced
		// default: false;
		bool _load_image_in_advanced;

		// load image in gray scale
		// default: false;
		bool _load_image_in_gray_scale;

		// image pair
		Camera _left_image;
		Camera _right_image;
	};
}