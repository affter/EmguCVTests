using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XFeatures2D;
using System;
using System.Drawing;
using System.IO;

namespace EmguCV
{
    class Program
    {
        static void Main(string[] args)
        {
            var observedPath = Path.GetFullPath("...\\...\\...\\source\\trinity2.jpg");
            var modelPath = Path.GetFullPath("...\\...\\...\\source\\trinity4.jpg");
            var observedMat = new Mat(observedPath, ImreadModes.Color);
            var modelMat = new Mat(modelPath, ImreadModes.Color);
            var result = Draw(modelPath, observedPath, modelMat, observedMat);
            CvInvoke.Imshow("result", result);
            CvInvoke.WaitKey();
        }

        public static void FindMatch(string modelFileName, string observedFileName, out VectorOfKeyPoint modelKeyPoints, out VectorOfKeyPoint observedKeyPoints, VectorOfVectorOfDMatch matches, out Mat mask)
        {
            int k = 2;
            double uniquenessThreshold = 0.8;

            modelKeyPoints = new VectorOfKeyPoint();
            observedKeyPoints = new VectorOfKeyPoint();
            {
                using (UMat uModelImage = CvInvoke.Imread(modelFileName, ImreadModes.Color).GetUMat(AccessType.Read))
                using (UMat uObservedImage = CvInvoke.Imread(observedFileName, ImreadModes.Color).GetUMat(AccessType.Read))
                {
                    SIFT sift = new SIFT();
                    UMat modelDescriptors = new UMat();
                    sift.DetectAndCompute(uModelImage, null, modelKeyPoints, modelDescriptors, false);
                    
                    UMat observedDescriptors = new UMat();
                    sift.DetectAndCompute(uObservedImage, null, observedKeyPoints, observedDescriptors, false);
                    BFMatcher matcher = new BFMatcher(DistanceType.L2);
                    matcher.Add(modelDescriptors);

                    matcher.KnnMatch(observedDescriptors, matches, k, null);
                    mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                    mask.SetTo(new MCvScalar(255));
                    Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);
                }
            }
        }

        public static Mat Draw(string modelFileName, string observedFileName, Mat modelImage, Mat observedImage)
        {
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            using (VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch())
            {
                Mat mask;
                FindMatch(modelFileName, observedFileName, out modelKeyPoints, out observedKeyPoints, matches,
                   out mask);

                //Draw the matched keypoints
                Mat result = new Mat();
                Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, observedImage, observedKeyPoints,
                   matches, result, new MCvScalar(255, 255, 255), new MCvScalar(255, 255, 255), mask);
                return result;

            }
        }
    }
}
