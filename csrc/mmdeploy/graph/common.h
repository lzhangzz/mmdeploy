// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_GRAPH_COMMON_H_
#define MMDEPLOY_SRC_GRAPH_COMMON_H_

#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/value.h"

namespace mmdeploy::graph {

namespace {

template <typename T>
inline auto Check(const T& v) -> decltype(!!v) {
  return !!v;
}

template <typename T>
inline std::true_type Check(T&&) {
  return {};
}

}  // namespace

template <typename EntryType, typename RetType = typename Creator<EntryType>::ReturnType>
inline Result<RetType> CreateFromRegistry(const Value& config, const char* key = "type") {
  MMDEPLOY_INFO("config: {}", config);
  auto type = config[key].get<std::string>();
  auto creator = Registry<EntryType>::Get().GetCreator(type);
  if (!creator) {
    MMDEPLOY_ERROR("failed to find module creator: {}", type);
    return Status(eEntryNotFound);
  }
  auto inst = creator->Create(config);
  if (!Check(inst)) {
    MMDEPLOY_ERROR("failed to create module: {}", type);
    return Status(eFail);
  }
  return std::move(inst);
}



Result<std::vector<std::string>> ParseStringArray(const Value& value);

template <typename BuilderType>
inline Result<std::unique_ptr<Node>> BuildFromConfig(Value config) {
  BuilderType builder(std::move(config));
  OUTCOME_TRY(builder.SetInputs());
  OUTCOME_TRY(builder.SetOutputs());
  return builder.Build();
}

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_SRC_GRAPH_COMMON_H_
