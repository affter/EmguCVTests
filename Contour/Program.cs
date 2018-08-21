using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Drawing;

namespace Contour
{
    class Program
    {
        static void Main(string[] args)
        {
            var img = new Image<Gray, byte>("../../../source/human.png");
            var top = img.ThresholdBinary(new Gray(174), new Gray(255));
            var bottom = img.ThresholdBinary(new Gray(175), new Gray(255));
            var binaryImg = top.Sub(bottom);
            CvInvoke.Erode(binaryImg, binaryImg, new Mat(), new Point(-1, -1), 1, BorderType.Default, new MCvScalar(255, 0, 0));
            CvInvoke.Dilate(binaryImg, binaryImg, new Mat(), new Point(-1, -1), 1, BorderType.Default, new MCvScalar(255, 0, 0));
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContourTree(binaryImg, contours, ChainApproxMethod.ChainApproxSimple);
            CvInvoke.DrawContours(img, contours, -1, new MCvScalar(255, 0, 0));
            CvInvoke.Imshow("White man", img);
            CvInvoke.WaitKey();
            //UMat edges = new UMat();
            //VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            //UMat newEdges = new UMat();
            //CvInvoke.Canny(img, edges, 20, 100, 3);
            //CvInvoke.Imshow("edges", edges);
            //CvInvoke.Dilate(edges, edges, new Mat(), new Point(-1, -1), 1, BorderType.Default, new MCvScalar(255, 0, 0));
            //CvInvoke.Erode(edges, edges, new Mat(), new Point(-1, -1), 1, BorderType.Default, new MCvScalar(255, 0, 0));
            //CvInvoke.FindContours(edges, contours, newEdges, RetrType.Tree, ChainApproxMethod.ChainApproxSimple);
            //CvInvoke.Canny(img, newEdges, 20, 180, 3);
            //CvInvoke.DrawContours(img, contours, -1, new MCvScalar(255, 0, 0));
            //CvInvoke.Imshow("edges", edges);
            //CvInvoke.Imshow("grayscale", img);
            //CvInvoke.WaitKey();
        }
    }
}
