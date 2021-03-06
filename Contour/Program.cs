﻿using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Linq;
using System.Drawing;
using System.Collections;
using System.Collections.Generic;

namespace Contour
{
    class Program
    {
        static void Main(string[] args)
        {
            var img = new Image<Bgr, byte>("../../../source/human.png");
            Image<Gray, Byte> grayImage = img.Convert<Gray, Byte>();
            int y = 0;
            var top = grayImage.ThresholdBinary(new Gray(174), new Gray(255));
            var bottom = grayImage.ThresholdBinary(new Gray(175), new Gray(255));
            var binaryImg = top.Sub(bottom);
            CvInvoke.Erode(binaryImg, binaryImg, new Mat(), new Point(-1, -1), 1, BorderType.Default, new MCvScalar(255, 0, 0));
            CvInvoke.Dilate(binaryImg, binaryImg, new Mat(), new Point(-1, -1), 1, BorderType.Default, new MCvScalar(255, 0, 0));
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContourTree(binaryImg, contours, ChainApproxMethod.ChainApproxSimple);
            CvInvoke.DrawContours(img, contours, -1, new MCvScalar(0, 0, 255));
            var imgForNormalsWithoutSharpEdges = img.Clone();
            CvInvoke.Imshow("Contour", img);
            for (int i = 0; i < contours[0].Size - 10; i += 10)
            {
                DrawNormal(img, contours[0][i], contours[0][i + 10]);
                if (i + 20 > contours[0].Size)
                {
                    DrawNormal(img, contours[0][i + 10], contours[0][0]);
                }
            }
            CvInvoke.Imshow("Normals", img);
            var normalBases = CalculateNormalBases(contours);
            foreach (var normalBase in normalBases)
            {
                DrawNormal(imgForNormalsWithoutSharpEdges, normalBase.Item1, normalBase.Item2); ;
            }
            CvInvoke.Imshow("ImageWithNormalsWithoutSharpEdges", imgForNormalsWithoutSharpEdges);
            CvInvoke.WaitKey();
        }

        private static void DrawNormal(CvArray<byte> img, Point point1, Point point2)
        {
            Point startPoint = Point.Empty;
            Point endPoint = Point.Empty;
            CalculateNormal(img, point1, point2, out startPoint, out endPoint);
            CvInvoke.Line(img, point1, point2, new MCvScalar(0, 255, 0));
            CvInvoke.Line(img, startPoint, endPoint, new MCvScalar(0, 255, 0));
        }

        private static IEnumerable<Tuple<Point, Point>> CalculateNormalBases(VectorOfVectorOfPoint contours, int coeff = 5)
        {
            var t = new List<Tuple<Point, Point>>();
            Point start = contours[0][0];
            Point end = contours[0][1];
            int diffX = 0, diffY = 0;
            for (int i = 1; i < contours[0].Size - 1; i++)
            {
                diffX = Math.Abs(end.X - start.X);
                diffY = Math.Abs(end.Y - start.Y);
                if (diffX < coeff && diffY < coeff)
                {
                    end = contours[0][++i];
                }
                else
                {
                    t.Add(new Tuple<Point, Point>(start, end));
                    start = end;
                    end = contours[0][++i];
                }
            }
            int j = contours[0].Size - 1;
            start = contours[0][0];
            end = contours[0][j];
            diffX = Math.Abs(end.X - start.X);
            diffY = Math.Abs(end.Y - start.Y);
            while (diffX < coeff && diffY < coeff)
            {
                end = contours[0][--j];
            }
            t.Add(new Tuple<Point, Point>(end, start));
            return t;
        }

        private static void CalculateNormal(CvArray<byte> img, Point point1, Point point2, out Point startPoint, out Point endPoint, int normalLength = 10)
        {
            int startX = (point1.X + point2.X) / 2;
            int startY = (point1.Y + point2.Y) / 2;
            double coeff = (double)(point2.Y - point1.Y) / (point2.X - point1.X);

            Console.WriteLine(coeff);
            int endX;
            if (point2.Y < point1.Y)
            {
                endX = (int)Math.Round(normalLength / Math.Sqrt(1 + 1 / (coeff * coeff)) + startX);
            }
            else
            {
                endX = (int)Math.Round(-normalLength / Math.Sqrt(1 + 1 / (coeff * coeff)) + startX);
            }
            int endY;
            if (coeff == 0)
            {
                if (point1.X > point2.X)
                {
                    endY = startY - normalLength;
                }
                else
                {
                    endY = startY + normalLength;
                }
            }
            else
            {
                endY = (int)Math.Round(startY + startX / coeff - endX / coeff);
            }
            startPoint = new Point(startX, startY);
            endPoint = new Point(endX, endY);
        }
    }
}
