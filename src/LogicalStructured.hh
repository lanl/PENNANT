/*
 * LogicalStructured.hh
 *
 *  Created on: Aug 16, 2016
 *      Author: jgraham
 */

#ifndef SRC_LOGICALSTRUCTURED_HH_
#define SRC_LOGICALSTRUCTURED_HH_


#include "LogicalUnstructured.hh"


class LogicalStructured : public LogicalUnstructured {
public:
    LogicalStructured(Context ctx, HighLevelRuntime *runtime);
    void allocate(int nElements);
    template <typename TYPE>
      TYPE* getRawPtr(FieldID FID);

};

#endif /* SRC_LOGICALSTRUCTURED_HH_ */
