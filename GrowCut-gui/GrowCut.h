#include <opencv/cv.h>

namespace ys {
	class GrowCut {
	public:
		GrowCut(
			const cv::Mat& src,
			const cv::Mat& label
			);
		int run_once();
		void run();

		const cv::Mat& getLabel();
		const cv::Mat& getStrength();
	private:
		cv::Mat _strength;
		cv::Mat _image;
		cv::Mat _label;

		cv::Mat _next_label;
		cv::Mat _next_strength;

	};
};