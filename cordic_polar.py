#!/usr/bin/python2

from operator import mul
import math
from math import degrees, atan, atan2, pi
import sys
import argparse

class Plotter(object):
    def __init__(self, filename, maxsize):
        global plt
        global PdfPages
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        import numpy as np
        self.maxsize = maxsize
        self.color='b'

        self.printed=False
        self.pdf = PdfPages(filename)

    def newpage(self, title):
        if self.printed:
            self._save()
        plt.figure(figsize=(6,6))
        plt.title(title)

    def _save(self):
        size = self.maxsize / 1.5 # a little more than sqrt(2)
        plt.axis([-size, size,
                  -size, size ])
        self.pdf.savefig()
        plt.close()
        self.printed=False

    def save(self):
        self._save()
        self.pdf.__exit__(1,2,3)

    def arrow(self, start, end):
        plt.plot((start[1],end[1]),(start[0],end[0]), color=self.color)
        self.printed=True

class CordicGen:
    def __init__(self, datatype, n, pdfname=None):
        if datatype=='int32_t':
            self.doubletype='int64_t'
            self.one_rev = 1<<32
            self.scale = (1<<31)-1
        elif datatype=='int16_t':
            self.doubletype='int32_t'
            self.one_rev = 1<<16
            self.scale = (1<<15)-1
        elif datatype=='int8_t':
            self.doubletype='int8_t'
            self.one_rev = 1<<8
            self.scale = (1<<7)-1
        else:
            raise KeyError("Unknown datatype %s"%datatype)

        self.datatype = datatype
        self.n = n
        self.name='"%s"'%" ".join(sys.argv)
        self.pdfname = pdfname

    def get_cordic_const(self):
        return round(self.scale * reduce(mul, [1/(1+4**-i)**0.5 for i in range(self.n)]))

    def get_cordic_range(self):
        return [round(self.one_rev*atan(2**-i)/(2*pi)) for i in range(self.n)]

    def get_c_function(self,size="int"):
        function = """#include <stdint.h>
#include <stdbool.h>

#ifndef ATAN_OFFSET
#define ATAN_OFFSET 0
#endif

//#define ARS(x,i) (x/(1u<<i))

// arithmetic right shift of signed integer
#ifndef ARS
#define ARS(x,i) (x >> i)
#endif

#ifndef REVOLUTION
#define REVOLUTION %(one_rev)d
#endif

static const %(type)s cordic[]={ %(constants)s };

static void rect_to_polar(%(type)s y, %(type)s x, %(type)s *angle, %(type)s *mag) {

   %(type)s theta;
   %(type)s i;

   theta=ATAN_OFFSET;

   /*
     this CORDIC valid for -pi..pi.  Map anything else to the proper
     quadrant.
   */
   if (x<0) {
      %(type)s old_y = y;
      y = x;
      x = old_y;

      if (old_y>0) {
          // move second quadrant into first (rotate -90 degrees)
          y = -y;
          theta += REVOLUTION/4;
      } else {
          // move third quadrant into forth (rotate +90 degrees)
          x = -x;
          theta -= REVOLUTION/4;
      }
   }

   for (i=0; i<%(n)d; i++) {
     %(type)s last_x;

     last_x=x;

     if (y<0) { // sign=1
       x -= ARS(y,i);
       y += ARS(last_x,i);
       theta -= cordic[i];
     } else {
       x += ARS(y,i);
       y -= ARS(last_x,i);
       theta += cordic[i];
     }
   }

   /* n=%(n)d
      scale=%(scale)d
      const=%(const)d
      constants=%(constants)s
   */

   if (mag) *mag=(%(double)s)x*%(const)d/%(scale)d;
   if (angle) *angle = theta %% REVOLUTION;
}

static void polar_to_rect(%(type)s angle, %(type)s mag, %(type)s *x_out, %(type)s *y_out) {


   %(type)s theta;
   %(type)s i;

   %(type)s x = (%(double)s)mag*%(const)d/%(scale)d;;
   %(type)s y = 0;

   theta=ATAN_OFFSET + angle;

   /* map to -pi..pi.

      If we go halfway around with theta, compensate by changing x 180
      degrees */

   if ((theta > REVOLUTION/4) ||
       (theta < -REVOLUTION/4)) {
      theta += REVOLUTION/2;
      x = -x;
   }

   for (i=0; i<%(n)d; i++) {
     %(type)s last_x;

     last_x=x;

     if (theta > 0) {
       x -= ARS(y,i);
       y += ARS(last_x,i);
       theta -= cordic[i];
     } else {
       x += ARS(y,i);
       y -= ARS(last_x,i);
       theta += cordic[i];
     }
   }

   if (x_out) *x_out = x;
   if (y_out) *y_out = y;
}

static %(type)s cordic_atan2(%(type)s y, %(type)s x) {
   %(type)s ret;
   rect_to_polar(y, x, &ret, NULL);
   return ret;
}

#define FULL_SCALE %(scale)d/2 // For sin() and cos()

static %(type)s cordic_sin(%(type)s x) {
   %(type)s ret;
   polar_to_rect(x, FULL_SCALE, &ret, NULL);
   return ret;
}

static %(type)s cordic_cos(%(type)s x) {
   %(type)s ret;
   polar_to_rect(x, FULL_SCALE, NULL, &ret);
   return ret;
}


"""%({'constants':", ".join(["%d"%r for r in self.get_cordic_range()]),
      'double':self.doubletype,
      'one_rev':self.one_rev,
      'n':self.n,
      'const':self.get_cordic_const(),
      'scale':self.scale,
      'type':size})

        return function

    def test_c_function(self, func, test_values):
        import subprocess
        import os,sys

        filename="/tmp/cordic_test_%d.out"%os.getpid()

        try:
            os.unlink(filename)
        except OSError: pass

        p=subprocess.Popen("cc -o "+filename+" -xc - ", stdin=subprocess.PIPE, shell=True)
        fd = p.stdin
        #fd = file('/tmp/this.c','w')

        print >>fd, "#include <stdio.h>"
        print >>fd, func

        print >>fd, """

void print_rect_polar(%(size)s x, %(size)s y) {
   %(size)s angle, mag;
   rect_to_polar(y, x, &angle, &mag);
   printf("rp %%d %%d %%d %%d\\n",x,y,angle,mag);
}

void print_polar_rect(%(size)s angle, %(size)s mag) {
   %(size)s x, y;
   polar_to_rect(angle, mag, &x, &y);
   printf(\"pr %%d %%d %%d %%d\\n",x,y,angle,mag);
}

int main(void) {
"""%{'size':self.datatype}

        for tv in test_values:
            print >>fd, "   print_rect_polar(%d, %d);"%tv
            ang,mag = self.native_rect_to_polar(tv[1],tv[0])
            print >>fd, "   print_polar_rect(%d, %d);"%(ang,mag)

        print >>fd, "   return 0;\n}"
        p.stdin.close()
        assert 0==p.wait()

        p=subprocess.Popen(filename, stdout=subprocess.PIPE, shell=True)

        ret=[line.split() for line in p.stdout]
        assert 0==p.wait()

        os.unlink(filename)
        return ret

    def get_test_case(self):
        import math
        ret=[]

        # grid
        r = int(self.scale/(2*math.sqrt(2)))
        s = r/20
        if s==0: s=1
        for x in range(-r,r,s):
            for y in range(-r,r,s):
                ret.append((y,x))

        # circle
        d=self.scale/2
        for theta in range(360):
            y=round(d*math.sin(theta*math.pi/180))
            x=round(d*math.cos(theta*math.pi/180))
            ret.append((y,x))

        # small circle
        d=self.scale/128
        for theta in range(360):
            y=round(d*math.sin(theta*math.pi/180))
            x=round(d*math.cos(theta*math.pi/180))
            ret.append((y,x))

        # spiral
        for theta,d in zip(range(360),range(self.scale / 1640,
                                            self.scale / 86)):
            y=round(d*math.sin(theta*math.pi/180))
            x=round(d*math.cos(theta*math.pi/180))
            ret.append((y,x))

        return ret

    def native_rect_to_polar(self,y,x):
        angle = math.atan2(y, x) * self.one_rev/(2*pi)
        magnitude = math.sqrt(x*x+y*y) # pop, pop!
        return angle,magnitude

    def get_tested_c_function(self):
        size = self.datatype

        def native_polar_to_rect(ang,mag):
            ang_rad = ang * 2*pi / self.one_rev

            y = mag * math.sin(ang_rad)
            x = mag * math.cos(ang_rad)

            return x,y

        function=self.get_c_function(size)

        test_cases = self.get_test_case()
        c=self.test_c_function(function,test_cases)

        rp_tests = [cl[1:] for cl in c if cl[0]=='rp']
        pr_tests = [cl[1:] for cl in c if cl[0]=='pr']

        assert(rp_tests)
        assert(pr_tests)

        if self.pdfname:
            self.plotter = Plotter(self.pdfname, self.scale)
            self.plotter.newpage('"%s" rect_to_polar()'%commandline)

        else:
            self.plotter = None

        #e=[]
        e1=[]
        mag_err=[]
        for rp_test in rp_tests:
            x,y,ang_out,mag_out = map(float, rp_test)
            ang_ref,mag_ref = self.native_rect_to_polar(y,x)

            err=ang_out - ang_ref

            while (err>(self.one_rev/2)): err -= self.one_rev;
            while (err<=(-self.one_rev/2)): err += self.one_rev;

            e1.append(err)

            mag_err.append(mag_ref - mag_out)

            if self.plotter:
                x_out, y_out = native_polar_to_rect(ang_out, mag_out)
                self.plotter.arrow((x,y),(x_out,y_out))

        preamble="""/*
   CORDIC rect_to_polar generated by %s

   result is one revolution per %d counts.

   rect_to_polar():
     Angle performance test (error as percent of revolution):
          Maximum: %+5.3f%%
             Mean:  %5.3f%%
          Minimum: %+5.3f%%
        mean(abs):  %5.3f%%
              rms:  %5.3f%%

     Magnitude test
          Maximum: %+5.3f
          Minimum: %+5.3f
              rms:  %5.3f

"""%(self.name,
     self.one_rev,
     100*max(e1)/(self.one_rev),
     100*sum(e1)/(len(e1)*self.one_rev),
     100*min(e1)/(self.one_rev),
     100*(sum([abs(e) for e in e1]))/(len(e1)*self.one_rev),
     100*math.sqrt(sum([e*e for e in e1]))/(len(e1)*self.one_rev),
     max(mag_err),
     min(mag_err),
     math.sqrt(sum([e*e for e in mag_err]))/(len(mag_err)))

        if self.plotter:
            self.plotter.newpage('"%s" polar_to_rect()'%commandline)
            self.plotter.color='r'

        #e=[]
        e1=[]
        x_err=[]
        y_err=[]
        for pr_test in pr_tests:
            x_out,y_out,ang,mag = map(float, pr_test)
            x_ref,y_ref = native_polar_to_rect(ang,mag)

            x_err.append(x_out-x_ref)
            y_err.append(y_out-y_ref)

            if self.plotter:
                self.plotter.arrow((x_ref,y_ref),(x_out,y_out))

        preamble+="""
   polar_to_rect():
     X, Y performance:
          Maximum: %+5.3f %+5.3f
             Mean:  %5.3f %5.3f
          Minimum: %+5.3f %+5.3f
        mean(abs):  %5.3f %5.3f
              rms:  %5.3f %5.3f
    */
"""%(max(x_err),max(y_err),
     sum(x_err)/len(x_err),sum(y_err)/len(y_err),
     min(x_err),min(y_err),
     (sum([abs(e) for e in x_err]))/len(x_err),
     (sum([abs(e) for e in y_err]))/len(y_err),
     math.sqrt(sum([e*e for e in x_err]))/len(x_err),
     math.sqrt(sum([e*e for e in y_err]))/len(y_err))

        if self.plotter:
            self.plotter.save()

        return preamble+"\n"+function

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdfname', help='Filename for error report')
    parser.add_argument('passes',type=int, help='number of CORDIC passes')
    parser.add_argument('datatype', help='datatype (one of int8_t, int16_t, int32_t)')
    args = parser.parse_args()

    n = args.passes
    while 1:
        cg=CordicGen(args.datatype, n, args.pdfname)
        if cg.get_cordic_range()[-1]: break
        n=n-1

    commandline = "%s %d %s"%(sys.argv[0], n, args.datatype)

    print cg.get_tested_c_function()
