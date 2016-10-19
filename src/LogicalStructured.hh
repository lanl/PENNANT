/*
 * LogicalStructured.hh
 *
 *  Created on: Aug 16, 2016
 *      Author: jgraham
 *
 * Copyright (c) 2016, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 *
 */

#ifndef SRC_LOGICALSTRUCTURED_HH_
#define SRC_LOGICALSTRUCTURED_HH_


#include "LogicalUnstructured.hh"


class LogicalStructured : public LogicalUnstructured {
public:
    LogicalStructured(Context ctx, HighLevelRuntime *runtime);
    LogicalStructured(Context ctx, HighLevelRuntime *runtime, PhysicalRegion pregion);
    void allocate(int nElements);
    int size() const { return nElements; }
    template <typename TYPE>
      TYPE* getRawPtr(FieldID FID);

private:
    int nElements;
};

#endif /* SRC_LOGICALSTRUCTURED_HH_ */
