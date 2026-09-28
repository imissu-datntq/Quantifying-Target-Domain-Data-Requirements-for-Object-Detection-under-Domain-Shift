"""Compile real upstream C logic; stub only CUDA allocation for CPU intervals."""
import shutil
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'external/MS-DAYOLO'


@unittest.skipUnless(shutil.which('gcc') and ROOT.is_dir(), 'C compiler/upstream required')
class NativeRegressionTests(unittest.TestCase):
    def test_research_printer_keeps_head_indices_pixels_and_empty_images(self):
        source = (ROOT / 'src/detector.c').read_text()
        self.assertTrue('void print_research_detections(' in source,
                      'Missing indexed research export printer')
        function = 'void print_research_detections(' + source.split(
            'void print_research_detections(', 1)[1].split('\nvoid validate_detector(', 1)[0]
        result = self.compile_run('#include "darknet.h"\n#include <stdio.h>\n#include <math.h>\n#include <stdlib.h>\n' + function + r'''
int main(void) {
    float probs[10]={.9,0,0,0,0,0,0,0,0,.8};
    detection d={0}; d.bbox=(box){20,30,10,20}; d.prob=probs;
    print_research_detections(stdout,0,&d,1,10,1280,720);
    print_research_detections(stdout,1,&d,0,10,1280,720);
    return 0;
}
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        first, empty = map(json.loads, result.stdout.splitlines())
        self.assertEqual(first['index'], 0)
        self.assertEqual(first['detections'][0][0], 0)
        self.assertEqual(first['detections'][1][0], 9)
        self.assertEqual(first['detections'][0][2:], [15,20,25,40])
        self.assertAlmostEqual(first['detections'][0][1], .9, places=6)
        self.assertEqual(empty, dict(index=1,width=1280,height=720,detections=[]))

    def test_da_batch_uses_source_truth_and_target_pixels_only(self):
        network_text = (ROOT / 'src/network.c').read_text()
        function = 'float da_train_network_waitkey' + network_text.split(
            'float da_train_network_waitkey', 1)[1].split('\nfloat train_network_batch', 1)[0]
        data_text = (ROOT / 'src/data.c').read_text()
        helpers = data_text[data_text.index('void get_next_batch(data'):]
        helpers = helpers[:helpers.index('\nvoid ', helpers.index('void get_next_batch_target') + 5)]
        code = r'''
#include "darknet.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static int calls=0, updates=0;
void *xcalloc(size_t n,size_t size) { return calloc(n,size); }
void wait_key_cv(int delay) { abort(); }
void update_network(network net) { updates++; }
float train_network_datum(network net,float *x,float *y) {
    assert(net.current_subdivision==calls);
    assert(x[0]==10+calls && x[1]==30+calls);
    assert(y[0]==50+calls);
    calls++;
    return 2;
}
''' + helpers + function + r'''
int main(void) {
    float s0=10,s1=11,t0=30,t1=31,y0=50,y1=51;
    float *sx[]={&s0,&s1}, *tx[]={&t0,&t1}, *sy[]={&y0,&y1};
    data source={0}, target={0};
    source.X=(matrix){.rows=2,.cols=1,.vals=sx};
    source.y=(matrix){.rows=2,.cols=1,.vals=sy};
    target.X=(matrix){.rows=2,.cols=1,.vals=tx};
    target.y=(matrix){.rows=2,.cols=1,.vals=NULL};
    network net={0};
    net.batch=2; net.no_dc=1;
    int dc=0; net.dc=&dc;
    float domain_loss=1; layer l={0}; l.da_loss=&domain_loss; net.layers=&l;
    int iteration=0; net.cur_iteration=&iteration;
    float loss=da_train_network_waitkey(net,source,target,0);
    assert(calls==2 && updates==1 && iteration==1 && loss==1);
    return 0;
}
'''
        result = self.compile_run(code)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # Mutation must be detected: injecting target truth cannot pass the test.
        mutant = code.replace('get_next_batch_target(d_t, batch/2, i*batch/2, X+(batch*d.X.cols/2));',
                              'get_next_batch(d_t, batch/2, i*batch/2, X+(batch*d.X.cols/2), y);')
        self.assertNotEqual(mutant, code)
        self.assertNotEqual(self.compile_run(mutant).returncode, 0)

    def compile_run(self, code, extra=()):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'test.c'
            binary = Path(folder) / 'test'
            source.write_text(code)
            build = subprocess.run(['gcc', '-w', '-ffunction-sections', '-fdata-sections',
                '-I' + str(ROOT / 'include'), '-I' + str(ROOT / 'src'), str(source),
                *map(str, extra), '-Wl,--gc-sections', '-pthread', '-lm', '-o', str(binary)],
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            return subprocess.run([str(binary)], cwd=folder, capture_output=True, text=True)

    def test_source_image_maps_to_separate_label_directory(self):
        result = self.compile_run(r'''
#include <stdio.h>
#include <string.h>
void replace_image_to_label(const char *, char *);
int main(void) {
    char out[4096];
    replace_image_to_label("/run/source/images/example.jpg", out);
    puts(out);
    return strcmp(out, "/run/source/labels/example.txt") != 0;
}
''', [ROOT / 'src/utils.c'])
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_new_pinned_block_does_not_overlap_next_allocation(self):
        text = (ROOT / 'src/dark_cuda.c').read_text()
        function = text.split('float *cuda_make_array_pinned_preallocated(float *x, size_t n)')[1]
        function = function.split('\nfloat *cuda_make_array_pinned(')[0]
        code = r'''
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaHostRegisterMapped 2
#define cudaMemcpyDefault 0
#define CHECK_CUDA(x) do { if(x) abort(); } while(0)
static int cudaHostAlloc(void **p, size_t n, int flags) { *p=malloc(n); return !*p; }
static int cudaMemcpyAsync(void *d, void *s, size_t n, int kind, int stream) { memcpy(d,s,n); return 0; }
static int get_cuda_stream(void) { return 0; }
static float **pinned_ptr=NULL;
static size_t pinned_num_of_blocks=0, pinned_index=0, pinned_block_id=0;
static const size_t pinned_block_size=8192;
static pthread_mutex_t mutex_pinned=PTHREAD_MUTEX_INITIALIZER;
float *cuda_make_array_pinned_preallocated(float *x, size_t n)
''' + function + r'''
int main(void) {
    float *a=cuda_make_array_pinned_preallocated(NULL,256);
    for(int i=0;i<256;i++) a[i]=17;
    float *b=cuda_make_array_pinned_preallocated(NULL,256);
    for(int i=0;i<256;i++) b[i]=29;
    int bad=0;
    for(int i=0;i<256;i++) if(a[i]!=17) bad=1;
    printf("first allocation preserved: %d\n", !bad);
    for(size_t i=0;i<pinned_num_of_blocks;i++) free(pinned_ptr[i]);
    free(pinned_ptr);
    return bad;
}
'''
        result = self.compile_run(code)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
