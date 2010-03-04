#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path

import kernel_generate_fromcsv

# ------------------------------------------------------------------------------
def kernel_generate_fromcsv_l(
    get_fvector_obj,
    output_dir,
    output_suffix,
    input_csv_fname_l,
    **kwargs):
    
    assert(path.isdir(output_dir))

    print len(input_csv_fname_l), "files to process"
    
    for input_csv_fname in input_csv_fname_l:
        output_fname = path.join(output_dir,
                                 path.basename(input_csv_fname) + output_suffix)
        print "output_fname:", output_fname
        kernel_generate_fromcsv.kernel_generate_fromcsv(
            input_csv_fname,
            output_fname,
            get_fvector_obj,
            **kwargs)


# ------------------------------------------------------------------------------
def main():

    parser = kernel_generate_fromcsv.get_optparser()
    parser.usage = ("usage: %prog [options] "
                    "<input_suffix> "
                    "<output_dir> "
                    "<output_suffix> "
                    "<input_csv_filename1> "
                    " ... "
                    "<input_csv_filenameN> "
                    )

    opts, args = parser.parse_args()

    if len(args) < 3:
        parser.print_help()
    else:
        input_suffix = args[0]
        output_dir = args[1]
        output_suffix = args[2]

        input_csv_fname_l = args[3:]

        options = vars(opts)
        input_path = options.pop('input_path')
        
        get_fvector_class = kernel_generate_fromcsv.GetFvectorFromSuffix
        get_fvector_obj = get_fvector_class(
            input_suffix,
            input_path = input_path,
            variable_name = opts.variable_name)

        kernel_generate_fromcsv_l(get_fvector_obj,
                                  output_dir,
                                  output_suffix,
                                  input_csv_fname_l,
                                  **options
                                  )

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
